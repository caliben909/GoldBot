"""
Order Manager - Handles order placement and management
"""
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)

class OrderManager:
    """
    Manages order lifecycle: placement, modification, cancellation
    """
    
    def __init__(self, config: dict, execution_engine):
        self.config = config
        self.execution_engine = execution_engine
        self.active_orders = {}
        self.order_history = []
        self.pending_cancels = set()
        
        logger.info("OrderManager initialized")
    
    async def place_order(self, signal: Dict) -> Optional[Dict]:
        """Place new order based on signal"""
        try:
            # Generate order ID
            order_id = str(uuid.uuid4())[:8]
            
            # Create order object
            order = {
                'id': order_id,
                'symbol': signal['symbol'],
                'direction': signal['direction'],
                'type': signal.get('order_type', 'MARKET'),
                'quantity': signal['position_size'],
                'price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'status': 'pending',
                'created_at': datetime.now(),
                'signal': signal
            }
            
            # Execute order based on platform
            if signal['symbol'].endswith(('USDT', 'BTC')):
                result = await self.execution_engine.execute_binance_trade(
                    symbol=signal['symbol'],
                    direction=signal['direction'],
                    position_size=signal['position_size'],
                    entry_price=signal['entry_price'],
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit']
                )
            else:
                result = await self.execution_engine.execute_mt5_trade(
                    symbol=signal['symbol'],
                    direction=signal['direction'],
                    position_size=signal['position_size'],
                    entry_price=signal['entry_price'],
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit']
                )
            
            if result and result.get('success'):
                order['status'] = 'filled'
                order['filled_at'] = datetime.now()
                order['external_id'] = result.get('order_id')
                order['fill_price'] = result.get('price', signal['entry_price'])
                
                self.active_orders[order_id] = order
                self.order_history.append(order)
                
                logger.info(f"Order {order_id} filled: {signal['direction']} {signal['symbol']}")
                return order
            else:
                order['status'] = 'failed'
                order['error'] = result.get('message', 'Unknown error')
                self.order_history.append(order)
                
                logger.error(f"Order {order_id} failed: {order['error']}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    async def modify_order(self, order_id: str, updates: Dict) -> bool:
        """Modify existing order"""
        if order_id not in self.active_orders:
            return False
        
        order = self.active_orders[order_id]
        
        try:
            # Modify stop loss
            if 'stop_loss' in updates:
                await self.execution_engine.modify_position(
                    symbol=order['symbol'],
                    order_id=order['external_id'],
                    stop_loss=updates['stop_loss']
                )
                order['stop_loss'] = updates['stop_loss']
            
            # Modify take profit
            if 'take_profit' in updates:
                await self.execution_engine.modify_position(
                    symbol=order['symbol'],
                    order_id=order['external_id'],
                    take_profit=updates['take_profit']
                )
                order['take_profit'] = updates['take_profit']
            
            order['updated_at'] = datetime.now()
            logger.info(f"Order {order_id} modified")
            return True
            
        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel open order"""
        if order_id not in self.active_orders:
            return False
        
        order = self.active_orders[order_id]
        
        try:
            result = await self.execution_engine.cancel_order(
                symbol=order['symbol'],
                order_id=order['external_id']
            )
            
            if result:
                order['status'] = 'cancelled'
                order['cancelled_at'] = datetime.now()
                del self.active_orders[order_id]
                logger.info(f"Order {order_id} cancelled")
                return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
        
        return False
    
    async def check_exit_conditions(self, position: Dict):
        """Check exit conditions for position"""
        try:
            # Breakeven after 1R
            if self._should_move_to_breakeven(position):
                await self.modify_order(position['order_id'], {
                    'stop_loss': position['entry_price']
                })
                logger.info(f"Moved {position['symbol']} to breakeven")
            
            # Partial close at 2R
            if self._should_partial_close(position):
                await self.execution_engine.close_position_partial(
                    symbol=position['symbol'],
                    percent=50
                )
                logger.info(f"Partially closed {position['symbol']} at 2R")
            
            # Trailing stop at 3R
            if self._should_activate_trailing(position):
                new_stop = self._calculate_trailing_stop(position)
                await self.modify_order(position['order_id'], {
                    'stop_loss': new_stop
                })
                logger.info(f"Activated trailing stop for {position['symbol']}")
                
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
    
    def _should_move_to_breakeven(self, position: Dict) -> bool:
        """Check if position should move to breakeven"""
        if position.get('breakeven_set'):
            return False
        
        risk = abs(position['entry_price'] - position['stop_loss'])
        profit = abs(position['current_price'] - position['entry_price'])
        
        return profit >= risk
    
    def _should_partial_close(self, position: Dict) -> bool:
        """Check if position should be partially closed"""
        if position.get('partial_closed'):
            return False
        
        risk = abs(position['entry_price'] - position['stop_loss'])
        profit = abs(position['current_price'] - position['entry_price'])
        
        return profit >= risk * 2
    
    def _should_activate_trailing(self, position: Dict) -> bool:
        """Check if trailing stop should be activated"""
        if position.get('trailing_active'):
            return False
        
        risk = abs(position['entry_price'] - position['stop_loss'])
        profit = abs(position['current_price'] - position['entry_price'])
        
        return profit >= risk * 3
    
    def _calculate_trailing_stop(self, position: Dict) -> float:
        """Calculate trailing stop price"""
        trail_distance = abs(position['entry_price'] - position['stop_loss']) * 0.5
        
        if position['direction'] == 'long':
            return position['current_price'] - trail_distance
        else:
            return position['current_price'] + trail_distance
    
    def get_active_orders(self) -> List[Dict]:
        """Get all active orders"""
        return list(self.active_orders.values())
    
    def get_order_history(self, limit: int = 100) -> List[Dict]:
        """Get order history"""
        return self.order_history[-limit:]