"""
Gold Institutional Quant Framework - Institutional-style trading system
(Macro + Session + AI + Adaptive Risk)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Market state data structure
class MarketState:
    """Represents current market conditions for Gold institutional trading"""
    
    def __init__(self, gold_price: float, gold_volume: float,
                 dxy_price: float, yield_10y: float,
                 spread: float, session: str, news_event: bool):
        self.gold_price = gold_price
        self.gold_volume = gold_volume
        self.dxy_price = dxy_price
        self.yield_10y = yield_10y
        self.spread = spread
        self.session = session
        self.news_event = news_event
        
    def __repr__(self):
        return (f"MarketState(gold={self.gold_price:.2f}, dxy={self.dxy_price:.2f}, "
                f"yield10y={self.yield_10y:.2f}, session={self.session}, news={self.news_event})")

# Session liquidity event types
class SessionEvent:
    """Represents a session liquidity event for Gold"""
    
    def __init__(self, timestamp: datetime, event_type: str,
                 asian_high: float, asian_low: float,
                 displacement_candle: bool, volume_spike: bool):
        self.timestamp = timestamp
        self.event_type = event_type  # "sweep_low" or "sweep_high"
        self.asian_high = asian_high
        self.asian_low = asian_low
        self.displacement_candle = displacement_candle
        self.volume_spike = volume_spike
        
    def __repr__(self):
        return (f"SessionEvent({self.timestamp.strftime('%H:%M')}, {self.event_type}, "
                f"spike={self.volume_spike}, displacement={self.displacement_candle})")

# ML prediction for trade success probability
class TradeProbability:
    """Represents the ML prediction for trade success probability"""
    
    def __init__(self, probability: float, features: List[float], confidence: float):
        self.probability = probability
        self.features = features
        self.confidence = confidence
        
    def __repr__(self):
        return f"TradeProbability(probability={self.probability:.2f}, confidence={self.confidence:.2f})"

class GoldInstitutionalFramework:
    """Gold Institutional Quant Framework implementation"""
    
    def __init__(self):
        self.initialized = False
        self.session_high = None
        self.session_low = None
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        logger.info("Gold Institutional Quant Framework initialized")
        
    def initialize(self) -> None:
        """Initialize the institutional framework"""
        self.initialized = True
        logger.info("Gold Institutional Quant Framework fully initialized and ready for trading")
        
    # ==================== MACRO BIAS ENGINE ====================
    
    def calculate_macro_score(self, dxy_trend: str, yield_trend: str, 
                              symbol_structure: str, symbol: str = "XAUUSD") -> int:
        """
        Calculate macro bias score based on institutional logic
        
        Args:
            dxy_trend: "up", "down", or "neutral"
            yield_trend: "up", "down", or "neutral"
            symbol_structure: "bullish", "bearish", or "neutral"
            symbol: Trading symbol (e.g., XAUUSD, EURUSD)
            
        Returns:
            Macro score (0-3)
        """
        score = 0
        
        # Correlation logic based on symbol
        if symbol == "XAUUSD":
            # Gold-specific correlation (DXY and 10Y yields)
            if (dxy_trend == "down" and symbol_structure == "bullish") or \
               (dxy_trend == "up" and symbol_structure == "bearish"):
                score += 1
                
            if (yield_trend == "down" and symbol_structure == "bullish") or \
               (yield_trend == "up" and symbol_structure == "bearish"):
                score += 1
        elif symbol == "EURUSD":
            # EURUSD-specific correlation (DXY and EURGBP)
            if (dxy_trend == "down" and symbol_structure == "bullish") or \
               (dxy_trend == "up" and symbol_structure == "bearish"):
                score += 1
                
            if (yield_trend == "up" and symbol_structure == "bullish") or \
               (yield_trend == "down" and symbol_structure == "bearish"):
                score += 1
        else:
            # Default correlation for other currencies (similar to EURUSD)
            if (dxy_trend == "down" and symbol_structure == "bullish") or \
               (dxy_trend == "up" and symbol_structure == "bearish"):
                score += 1
                
            if (yield_trend == "up" and symbol_structure == "bullish") or \
               (yield_trend == "down" and symbol_structure == "bearish"):
                score += 1
        
        # Structure strength
        if symbol_structure != "neutral":
            score += 1
            
        return score
        
    def determine_macro_bias(self, score: int, symbol_structure: str) -> str:
        """
        Determine macro bias based on score and symbol structure
        
        Args:
            score: Macro score (0-3)
            symbol_structure: "bullish", "bearish", or "neutral"
            
        Returns:
            Bias classification: "strong_bullish", "moderate_bullish", 
            "strong_bearish", "moderate_bearish", or "neutral"
        """
        if score == 0:
            return "neutral"
            
        if symbol_structure == "bullish":
            if score == 3:
                return "strong_bullish"
            elif score == 2:
                return "moderate_bullish"
            elif score == 1:
                return "moderate_bullish"  # Still bullish but with lower conviction
        elif symbol_structure == "bearish":
            if score == 3:
                return "strong_bearish"
            elif score == 2:
                return "moderate_bearish"
            elif score == 1:
                return "moderate_bearish"  # Still bearish but with lower conviction
        
        return "neutral"
        
    # ==================== SESSION LIQUIDITY ENGINE ====================
    
    def identify_session_events(self, df: pd.DataFrame) -> Optional[SessionEvent]:
        """
        Identify session liquidity events based on Gold's session structure
        
        Args:
            df: Historical price data with session information
            
        Returns:
            SessionEvent if detected, None otherwise
        """
        if len(df) < 24:  # Need at least 12 hours of data
            return None
            
        # Get latest session data
        latest_price = df['close'].iloc[-1]
        latest_volume = df['volume'].iloc[-1]
        
        # Identify session boundaries
        current_time = df.index[-1].time()
        
        # Simplified session detection - recognize more opportunities
        if pd.Timestamp('00:00:00').time() <= current_time <= pd.Timestamp('16:00:00').time():
            # For Asian and London sessions, use rolling 8-hour range as reference
            lookback_period = 8 * 4  # 8 hours * 4 (15-minute candles)
            if len(df) >= lookback_period:
                reference_range = df[-lookback_period:]
                ref_high = reference_range['high'].max()
                ref_low = reference_range['low'].min()
                
                # Check for potential liquidity event
                if latest_price <= ref_low * 1.001 or latest_price >= ref_high * 0.999:
                    volume_spike = latest_volume > df['volume'].rolling(10).mean().iloc[-1] * 1.5
                    displacement = abs(latest_price - df['close'].iloc[-2]) > df['close'].iloc[-2] * 0.002
                    return SessionEvent(df.index[-1], "potential_sweep", ref_high, ref_low, displacement, volume_spike)
                    
        return None
        
    # ==================== AI CONFIDENCE SCORING ====================
    
    def build_feature_vector(self, df: pd.DataFrame, macro_score: int, 
                           session_alignment: bool, volume_strength: float,
                           orderflow_imbalance: float, spread_score: float,
                           news_penalty: int) -> List[float]:
        """
        Build feature vector for trade scoring
        
        Args:
            df: Historical price data
            macro_score: Macro score (0-3)
            session_alignment: Whether session aligns with macro
            volume_strength: Volume spike ratio
            orderflow_imbalance: Order flow imbalance score
            spread_score: Spread condition score
            news_penalty: News event penalty
            
        Returns:
            Feature vector for scoring
        """
        features = []
        
        # Macro features
        features.append(macro_score)
        
        # Session features
        features.append(1 if session_alignment else 0)
        
        # Volume features
        features.append(volume_strength)
        
        # Order flow features
        features.append(orderflow_imbalance)
        
        # Spread features
        features.append(spread_score)
        
        # News features
        features.append(news_penalty)
        
        return features
        
    def calculate_confidence_score(self, features: List[float]) -> float:
        """
        Calculate confidence score using weighted scoring
        
        Args:
            features: Feature vector
            
        Returns:
            Confidence score between 0-10
        """
        # Weighting scheme
        weights = [2.0, 1.5, 1.0, 1.2, 0.8, -1.5]
        
        score = sum(f * w for f, w in zip(features, weights))
        
        # Normalize to 0-10 range
        score = max(0, min(10, score))
        
        return round(score, 2)
        
    # ==================== ADAPTIVE RISK ENGINE ====================
    
    def calculate_adaptive_risk(self, confidence: float, equity: float, 
                             drawdown: float, trade_count: int, 
                             consecutive_losses: int) -> float:
        """
        Calculate adaptive position sizing based on institutional logic
        
        Args:
            confidence: Trade confidence score
            equity: Current equity
            drawdown: Current drawdown percentage
            trade_count: Total trade count
            consecutive_losses: Number of consecutive losses
            
        Returns:
            Position size in USD
        """
        base_risk = 0.015
        
        # Adjust risk based on drawdown
        if drawdown > 5:
            base_risk *= 0.5
            
        # Adjust risk based on consecutive losses
        if consecutive_losses >= 3:
            base_risk *= 0.8
            
        # Adjust based on confidence and winning streaks
        if confidence >= 9.0:
            return equity * 0.03
        elif confidence >= 8.0:
            return equity * 0.02
        elif confidence >= 6.0:
            return equity * base_risk
        elif confidence >= 4.0:
            return equity * 0.01
            
        return 0
        
    # ==================== TRADE EXECUTION ====================
    
    def calculate_position_size(self, risk_amount: float, entry_price: float,
                              stop_loss: float, lot_size: float = 100) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            risk_amount: Risk amount in USD
            entry_price: Entry price
            stop_loss: Stop loss price
            lot_size: Contract size per lot
            
        Returns:
            Position size in lots
        """
        if entry_price == stop_loss:
            return 0
            
        risk_per_lot = abs(entry_price - stop_loss) * lot_size
        
        if risk_per_lot > 0:
            return risk_amount / risk_per_lot
            
        return 0
        
    def calculate_entry_level(self, df: pd.DataFrame, event: SessionEvent, 
                            bias: str) -> float:
        """
        Calculate optimal entry level based on institutional logic
        
        Args:
            df: Historical price data
            event: Session liquidity event
            bias: Macro bias
            
        Returns:
            Entry price
        """
        latest_price = df['close'].iloc[-1]
        
        # Wait for retracement to order block or imbalance
        if bias == "strong_bullish" or bias == "moderate_bullish":
            return latest_price * 0.998
        elif bias == "strong_bearish" or bias == "moderate_bearish":
            return latest_price * 1.002
            
        return latest_price
        
    def calculate_stop_loss(self, df: pd.DataFrame, event: SessionEvent, 
                          bias: str) -> float:
        """
        Calculate stop loss based on institutional logic
        
        Args:
            df: Historical price data
            event: Session liquidity event
            bias: Macro bias
            
        Returns:
            Stop loss price
        """
        latest_price = df['close'].iloc[-1]
        
        if bias == "strong_bullish" or bias == "moderate_bullish":
            # Below liquidity sweep - tighter stop loss for better position sizing
            return min(event.asian_low * 0.998, latest_price * 0.995)
        elif bias == "strong_bearish" or bias == "moderate_bearish":
            # Above liquidity sweep - tighter stop loss for better position sizing
            return max(event.asian_high * 1.002, latest_price * 1.005)
            
        return latest_price
        
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                           risk_reward: float = 3.0) -> float:
        """
        Calculate take profit based on risk-reward ratio
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_reward: Risk-reward ratio (default: 2.0)
            
        Returns:
            Take profit price
        """
        risk = abs(entry_price - stop_loss)
        if entry_price > stop_loss:  # Long position
            return entry_price + (risk * risk_reward)
        else:  # Short position
            return entry_price - (risk * risk_reward)
            
    # ==================== TRADE MANAGEMENT ====================
    
    def calculate_trailing_stop(self, current_price: float, entry_price: float,
                             direction: str, initial_stop: float) -> float:
        """
        Calculate trailing stop based on institutional logic
        
        Args:
            current_price: Current price
            entry_price: Entry price
            direction: "long" or "short"
            initial_stop: Initial stop loss
            
        Returns:
            Trailing stop price
        """
        # Trail using 20 EMA approach
        if direction == "long":
            profit = current_price - entry_price
            if profit >= 0.02 * entry_price:  # 2% profit
                return max(initial_stop, current_price * 0.98)
        else:
            profit = entry_price - current_price
            if profit >= 0.02 * entry_price:  # 2% profit
                return min(initial_stop, current_price * 1.02)
                
        return initial_stop
        
    def should_exit_trade(self, current_price: float, take_profit: float,
                        stop_loss: float, direction: str) -> str:
        """
        Determine if trade should exit
        
        Args:
            current_price: Current price
            take_profit: Take profit price
            stop_loss: Stop loss price
            direction: "long" or "short"
            
        Returns:
            Exit reason: "take_profit", "stop_loss", or "none"
        """
        if direction == "long":
            if current_price >= take_profit:
                return "take_profit"
            elif current_price <= stop_loss:
                return "stop_loss"
        else:
            if current_price <= take_profit:
                return "take_profit"
            elif current_price >= stop_loss:
                return "stop_loss"
                
        return "none"
        
    # ==================== FULL STRATEGY EXECUTION ====================
    
    def execute_strategy(self, df: pd.DataFrame, equity: float, 
                       dxy_data: pd.DataFrame, yield_data: pd.DataFrame,
                       spread: float, news_event: bool, symbol: str = "XAUUSD") -> Dict:
        """
        Execute the full institutional strategy

        Args:
            df: Gold price data
            equity: Current equity
            dxy_data: DXY price data
            yield_data: 10-year Treasury yield data
            spread: Current spread
            news_event: Whether news event is pending

        Returns:
            Strategy execution result
        """
        result = {
            'should_trade': False,
            'direction': None,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'position_size': 0,
            'confidence_score': 0,
            'macro_score': 0,
            'session_event': None,
            'risk_amount': 0,
            'entry_reason': '',
            'risk_reward_ratio': 0
        }
        
        try:
            # Skip if news event is pending
            if news_event:
                logger.info("Skipping trade due to upcoming news event")
                return result
                
            # Skip if spread is too wide
            if spread > 0.5:
                logger.info(f"Spread too wide: {spread:.2f}, skipping trade")
                return result
                
            # Determine current session
            current_time = df.index[-1].time()
            session = "asian"
            if pd.Timestamp('08:00:00').time() <= current_time <= pd.Timestamp('16:00:00').time():
                session = "london"
            elif pd.Timestamp('16:00:00').time() <= current_time < pd.Timestamp('23:59:59').time():
                session = "ny"
            
            # Calculate macro score from actual data
            # Determine DXY trend
            if len(dxy_data) > 5:
                dxy_trend = "up" if dxy_data['close'].iloc[-1] > dxy_data['close'].iloc[-5] else "down"
            else:
                dxy_trend = "neutral"
                
            # Determine yield trend
            if len(yield_data) > 5:
                yield_trend = "up" if yield_data['close'].iloc[-1] > yield_data['close'].iloc[-5] else "down"
            else:
                yield_trend = "neutral"
                
            # Determine gold structure
            if len(df) > 10:
                gold_structure = "bullish" if df['close'].iloc[-1] > df['close'].iloc[-10] else "bearish"
            else:
                gold_structure = "neutral"
                
            macro_score = self.calculate_macro_score(dxy_trend, yield_trend, gold_structure, symbol)
            result['macro_score'] = macro_score
            
            logger.debug(f"Debug info - Symbol: {symbol}, Structure: {gold_structure}, DXY Trend: {dxy_trend}, Yield Trend: {yield_trend}, Macro Score: {macro_score}")
            
            # Skip only if there's no macro bias at all
            if macro_score < 0.5:
                logger.info(f"Macro score too low: {macro_score}, skipping trade")
                return result
                
            # Identify session liquidity event
            session_event = self.identify_session_events(df)
            result['session_event'] = session_event
            
            if not session_event:
                logger.debug("No session liquidity event detected")
                return result
                
            # Calculate confidence score
            features = self.build_feature_vector(df, macro_score, True, 1.5, 0.8, 0.9, 0)
            confidence = self.calculate_confidence_score(features)
            result['confidence_score'] = confidence
            
            # Higher confidence threshold to reduce losing trades
            min_confidence = 7.0
            if session == "ny":
                min_confidence = 8.0
                
            if confidence < min_confidence:
                logger.debug(f"Confidence score too low: {confidence:.2f}, skipping trade")
                return result
                
            # Determine direction based on macro bias
            macro_bias = self.determine_macro_bias(macro_score, gold_structure)
            if macro_bias in ["strong_bullish", "moderate_bullish"]:
                direction = "long"
            elif macro_bias in ["strong_bearish", "moderate_bearish"]:
                direction = "short"
            else:
                logger.debug(f"Neutral macro bias: {macro_bias}, skipping trade")
                return result
                
            result['direction'] = direction
            
            # Calculate entry and exit levels
            entry_price = self.calculate_entry_level(df, session_event, macro_bias)
            stop_loss = self.calculate_stop_loss(df, session_event, macro_bias)
            
            # Adjust risk-reward based on session
            risk_reward = 2.5
            if session == "ny":
                risk_reward = 2.0
                
            take_profit = self.calculate_take_profit(entry_price, stop_loss, risk_reward)
            
            # Calculate risk management - smaller risk for NY session
            drawdown = 0
            consecutive_losses = 0
            risk_amount = self.calculate_adaptive_risk(confidence, equity, drawdown,
                                                     self.trade_count, consecutive_losses)
            
            if session == "ny":
                risk_amount *= 0.7  # Reduce risk by 30% for NY session
            
            position_size = self.calculate_position_size(risk_amount, entry_price, stop_loss)
            
            if position_size > 0:
                result['should_trade'] = True
                result['entry_price'] = entry_price
                result['stop_loss'] = stop_loss
                result['take_profit'] = take_profit
                result['position_size'] = position_size
                result['risk_amount'] = risk_amount
                result['entry_reason'] = f"Macro aligned + Session sweep ({session})"
                result['risk_reward_ratio'] = risk_reward
                
                logger.info(f"Trade signal generated: {direction} at {entry_price:.2f}, "
                           f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}, "
                           f"Confidence: {confidence:.2f} (Session: {session})")
                
        except Exception as e:
            logger.error(f"Strategy execution error: {str(e)}")
            
        return result
        
    # ==================== PERFORMANCE TRACKING ====================
    
    def track_trade_result(self, trade_result: Dict) -> None:
        """Track trade performance metrics"""
        self.trade_count += 1
        
        if trade_result.get('profit') > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
            
        logger.info(f"Trade {self.trade_count} closed: Profit {trade_result.get('profit', 0):.2f}")
        
    def get_performance_summary(self) -> Dict:
        """Get trading performance summary"""
        if self.trade_count > 0:
            win_rate = self.win_count / self.trade_count
            loss_rate = self.loss_count / self.trade_count
        else:
            win_rate = 0
            loss_rate = 0
            
        return {
            'total_trades': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'loss_rate': loss_rate
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize the institutional framework
    framework = GoldInstitutionalFramework()
    framework.initialize()
    
    # Create dummy data
    dates = pd.date_range('2024-01-01', periods=100, freq='15T')
    df = pd.DataFrame({
        'open': [2000 + i * 0.1 for i in range(100)],
        'high': [2001 + i * 0.1 for i in range(100)],
        'low': [1999 + i * 0.1 for i in range(100)],
        'close': [2000 + i * 0.1 for i in range(100)],
        'volume': [10000 + i * 100 for i in range(100)]
    }, index=dates)
    
    dxy_data = pd.DataFrame({
        'close': [103 - i * 0.1 for i in range(100)]
    }, index=dates)
    
    yield_data = pd.DataFrame({
        'close': [4.2 - i * 0.01 for i in range(100)]
    }, index=dates)
    
    equity = 10000
    
    # Execute strategy
    result = framework.execute_strategy(df, equity, dxy_data, yield_data, 0.3, False)
    
    # Print result
    print("\nStrategy Result:")
    for key, value in result.items():
        if value is not None:
            if key in ['entry_price', 'stop_loss', 'take_profit']:
                print(f"{key}: {value:.2f}")
            elif key == 'risk_reward_ratio':
                print(f"{key}: {value:.1f}")
            elif key == 'should_trade':
                print(f"{key}: {'Yes' if value else 'No'}")
            else:
                print(f"{key}: {value}")
    
    # Get performance
    performance = framework.get_performance_summary()
    print("\nPerformance:")
    for key, value in performance.items():
        if key in ['win_rate', 'loss_rate']:
            print(f"{key}: {value:.1%}")
        else:
            print(f"{key}: {value}")
