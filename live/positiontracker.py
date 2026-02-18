"""
Position Tracker - Tracks open positions and P&L
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class PositionTracker:
    """
    Tracks all open positions and their P&L
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.positions = {}
        self.position_history = []
        self.daily_pnl = 0
        self.weekly_pnl = 0
        self.monthly_pnl = 0
        
        logger.info("PositionTracker initialized")
    
    async def add_position(self, order: Dict):
        """Add new position"""
        position = {
            'order_id': order['id'],
            'symbol': order['symbol'],
            'direction': order['direction'],
            'entry_price': order.get('fill_price', order['price']),
            'current_price': order.get('fill_price', order['price']),
            'quantity': order['quantity'],
            'stop_loss': order['stop_loss'],
            'take_profit': order['take_profit'],
            'open_time': datetime.now(),
            'unrealized_pnl': 0,
            'realized_pnl': 0,
            'status': 'open',
            'breakeven_set': False,
            'partial_closed': False,
            'trailing_active': False
        }
        
        self.positions[order['id']] = position
        logger.info(f"Position added: {order['symbol']} {order['direction']}")
    
    async def update_position(self, order_id: str, current_price: float):
        """Update position with current price"""
        if order_id not in self.positions:
            return
        
        position = self.positions[order_id]
        position['current_price'] = current_price
        
        # Calculate unrealized P&L
        if position['direction'] == 'long':
            position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
        else:
            position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['quantity']
        
        # Update daily/weekly/monthly P&L
        self._update_pnl_metrics(position['unrealized_pnl'])
    
    def _update_pnl_metrics(self, pnl: float):
        """Update P&L metrics"""
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.monthly_pnl += pnl
    
    async def close_position(self, order_id: str, exit_price: float, exit_reason: str):
        """Close position"""
        if order_id not in self.positions:
            return
        
        position = self.positions[order_id]
        
        # Calculate realized P&L
        if position['direction'] == 'long':
            realized_pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            realized_pnl = (position['entry_price'] - exit_price) * position['quantity']
        
        # Update position
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now()
        position['realized_pnl'] = realized_pnl
        position['exit_reason'] = exit_reason
        position['status'] = 'closed'
        
        # Move to history
        self.position_history.append(position)
        del self.positions[order_id]
        
        logger.info(f"Position closed: {position['symbol']} P&L: ${realized_pnl:.2f}")
    
    async def has_position(self, symbol: str) -> bool:
        """Check if position exists for symbol"""
        return any(p['symbol'] == symbol and p['status'] == 'open' 
                  for p in self.positions.values())
    
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for symbol"""
        for position in self.positions.values():
            if position['symbol'] == symbol and position['status'] == 'open':
                return position
        return None
    
    async def get_all_positions(self) -> List[Dict]:
        """Get all open positions"""
        return list(self.positions.values())
    
    async def get_position_count(self) -> int:
        """Get number of open positions"""
        return len(self.positions)
    
    async def get_total_exposure(self) -> float:
        """Get total position exposure"""
        total = 0
        for position in self.positions.values():
            total += position['quantity'] * position['current_price']
        return total
    
    async def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L"""
        total = 0
        for position in self.positions.values():
            total += position['unrealized_pnl']
        return total
    
    def get_daily_pnl(self) -> float:
        """Get daily P&L"""
        return self.daily_pnl
    
    def reset_daily_pnl(self):
        """Reset daily P&L"""
        self.daily_pnl = 0
    
    def get_position_summary(self) -> Dict:
        """Get position summary"""
        return {
            'total_positions': len(self.positions),
            'total_exposure': self.get_total_exposure(),
            'unrealized_pnl': self.get_unrealized_pnl(),
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'monthly_pnl': self.monthly_pnl
        }