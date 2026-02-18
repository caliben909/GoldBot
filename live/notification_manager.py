"""
Notification Manager - Multi-channel alert system
"""
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import aiohttp
import telegram
from twilio.rest import Client as TwilioClient
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import requests

logger = logging.getLogger(__name__)


class NotificationManager:
    """
    Multi-channel notification system
    
    Supports:
    - Telegram
    - Email
    - Slack
    - Discord
    - SMS (Twilio)
    - Webhook
    """
    
    def __init__(self, config: dict):
        self.config = config['user_experience']['alerts']
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients
        self.telegram_bot = None
        self.twilio_client = None
        self.session = None
        
        # Alert queues
        self.alert_queue = asyncio.Queue()
        self.priority_queue = asyncio.Queue()
        
        # Rate limiting
        self.last_sent = {}
        self.rate_limits = {
            'telegram': 20,  # messages per minute
            'email': 10,
            'sms': 5,
            'slack': 50
        }
        
        logger.info("NotificationManager initialized")
    
    async def initialize(self):
        """Initialize notification clients"""
        # Telegram
        if self.config['telegram']['enabled']:
            self.telegram_bot = telegram.Bot(
                token=self.config['telegram']['bot_token']
            )
            logger.info("Telegram bot initialized")
        
        # Twilio SMS
        if self.config['sms']['enabled']:
            self.twilio_client = TwilioClient(
                self.config['sms']['account_sid'],
                self.config['sms']['auth_token']
            )
            logger.info("Twilio client initialized")
        
        # HTTP session
        self.session = aiohttp.ClientSession()
        
        # Start background processor
        asyncio.create_task(self._process_alerts())
    
    async def send_trade_notification(self, trade: Dict):
        """Send trade execution notification"""
        message = self._format_trade_message(trade)
        
        await self.send_alert(
            message=message,
            priority='high',
            channels=['telegram', 'email'] if trade.get('profit', 0) > 1000 else ['telegram']
        )
    
    async def send_risk_alert(self, alert: Dict):
        """Send risk alert"""
        message = self._format_risk_message(alert)
        
        await self.send_alert(
            message=message,
            priority='critical',
            channels=['telegram', 'email', 'sms']
        )
    
    async def send_daily_summary(self, summary: Dict):
        """Send daily trading summary"""
        message = self._format_summary_message(summary)
        
        await self.send_alert(
            message=message,
            priority='normal',
            channels=['telegram', 'email']
        )
    
    async def send_error_alert(self, error: str, details: Dict = None):
        """Send error alert"""
        message = f"❌ ERROR: {error}\n"
        if details:
            message += f"Details: {json.dumps(details, indent=2)}"
        
        await self.send_alert(
            message=message,
            priority='critical',
            channels=['telegram', 'email']
        )
    
    async def send_startup_notification(self, info: Dict):
        """Send bot startup notification"""
        message = f"""
🚀 Trading Bot Started
═════════════════════
Time: {info.get('start_time')}
Account: {info.get('account')}
Balance: ${info.get('balance', 0):.2f}
Symbols: {', '.join(info.get('symbols', []))}
Mode: {info.get('mode', 'live')}
        """
        
        await self.send_alert(
            message=message,
            priority='normal',
            channels=['telegram', 'email']
        )
    
    async def send_shutdown_notification(self, info: Dict):
        """Send bot shutdown notification"""
        message = f"""
🛑 Trading Bot Shutdown
══════════════════════
Runtime: {info.get('runtime')}
Final P&L: ${info.get('final_pnl', 0):.2f}
Trades: {info.get('trades', 0)}
        """
        
        await self.send_alert(
            message=message,
            priority='normal',
            channels=['telegram', 'email']
        )
    
    async def send_alert(self, message: str, priority: str = 'normal',
                        channels: List[str] = None):
        """Send alert through specified channels"""
        if channels is None:
            channels = ['telegram']
        
        alert = {
            'message': message,
            'priority': priority,
            'channels': channels,
            'timestamp': datetime.now().isoformat()
        }
        
        if priority == 'critical':
            await self.priority_queue.put(alert)
        else:
            await self.alert_queue.put(alert)
    
    async def _process_alerts(self):
        """Process alert queue"""
        while True:
            try:
                # Process priority queue first
                if not self.priority_queue.empty():
                    alert = await self.priority_queue.get()
                    await self._send_to_channels(alert)
                
                # Process normal queue
                elif not self.alert_queue.empty():
                    alert = await self.alert_queue.get()
                    await self._send_to_channels(alert)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(1)
    
    async def _send_to_channels(self, alert: Dict):
        """Send alert to specified channels"""
        for channel in alert['channels']:
            try:
                if channel == 'telegram':
                    await self._send_telegram(alert['message'])
                elif channel == 'email':
                    await self._send_email(alert['message'])
                elif channel == 'sms':
                    await self._send_sms(alert['message'])
                elif channel == 'slack':
                    await self._send_slack(alert['message'])
                elif channel == 'discord':
                    await self._send_discord(alert['message'])
                
                # Rate limiting
                await self._apply_rate_limit(channel)
                
            except Exception as e:
                logger.error(f"Failed to send {channel} alert: {e}")
    
    async def _send_telegram(self, message: str):
        """Send Telegram message"""
        if not self.telegram_bot:
            return
        
        try:
            await self.telegram_bot.send_message(
                chat_id=self.config['telegram']['chat_id'],
                text=message[:4096],  # Telegram limit
                parse_mode='HTML'
            )
            logger.debug("Telegram message sent")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
    
    async def _send_email(self, message: str):
        """Send email"""
        if not self.config['email']['enabled']:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['username']
            msg['To'] = ', '.join(self.config['email']['recipients'])
            msg['Subject'] = f"Trading Bot Alert - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(
                self.config['email']['smtp_server'],
                self.config['email']['smtp_port']
            )
            server.starttls()
            server.login(
                self.config['email']['username'],
                self.config['email']['password']
            )
            server.send_message(msg)
            server.quit()
            
            logger.debug("Email sent")
            
        except Exception as e:
            logger.error(f"Email send failed: {e}")
    
    async def _send_sms(self, message: str):
        """Send SMS via Twilio"""
        if not self.twilio_client:
            return
        
        try:
            message = self.twilio_client.messages.create(
                body=message[:160],  # SMS limit
                from_=self.config['sms']['phone_number'],
                to=self.config['sms']['recipient_number']
            )
            logger.debug("SMS sent")
            
        except Exception as e:
            logger.error(f"SMS send failed: {e}")
    
    async def _send_slack(self, message: str):
        """Send Slack message"""
        if not self.config['slack']['enabled']:
            return
        
        try:
            payload = {
                'text': message,
                'username': 'Trading Bot',
                'icon_emoji': ':chart_with_upwards_trend:'
            }
            
            async with self.session.post(
                self.config['slack']['webhook_url'],
                json=payload
            ) as response:
                if response.status != 200:
                    logger.error(f"Slack send failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
    
    async def _send_discord(self, message: str):
        """Send Discord message"""
        if not self.config['discord']['enabled']:
            return
        
        try:
            payload = {'content': message}
            
            async with self.session.post(
                self.config['discord']['webhook_url'],
                json=payload
            ) as response:
                if response.status != 204:
                    logger.error(f"Discord send failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Discord send failed: {e}")
    
    async def _apply_rate_limit(self, channel: str):
        """Apply rate limiting"""
        limit = self.rate_limits.get(channel, 10)
        delay = 60 / limit  # seconds between messages
        
        last = self.last_sent.get(channel, 0)
        elapsed = time.time() - last
        
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        
        self.last_sent[channel] = time.time()
    
    def _format_trade_message(self, trade: Dict) -> str:
        """Format trade notification"""
        emoji = '✅' if trade.get('profit', 0) > 0 else '❌'
        
        return f"""
{emoji} Trade {trade.get('status', 'EXECUTED')}
═════════════════════
Symbol: {trade.get('symbol')}
Direction: {trade.get('direction', '').upper()}
Entry: ${trade.get('entry_price', 0):.4f}
Exit: ${trade.get('exit_price', 0):.4f}
Quantity: {trade.get('quantity', 0):.4f}
Profit: ${trade.get('profit', 0):.2f}
Pips: {trade.get('profit_pips', 0):.1f}
R Multiple: {trade.get('r_multiple', 0):.2f}
Time: {trade.get('exit_time', datetime.now())}
        """
    
    def _format_risk_message(self, alert: Dict) -> str:
        """Format risk alert"""
        return f"""
⚠️ RISK ALERT
═════════════
Type: {alert.get('type')}
Value: {alert.get('value', 0):.2f}%
Limit: {alert.get('limit', 0):.2f}%
Action: {alert.get('action', 'MONITOR')}
Time: {datetime.now()}
        """
    
    def _format_summary_message(self, summary: Dict) -> str:
        """Format daily summary"""
        return f"""
📊 Daily Trading Summary
═══════════════════════
Date: {summary.get('date')}
Total Trades: {summary.get('total_trades', 0)}
Win Rate: {summary.get('win_rate', 0):.1f}%
Total P&L: ${summary.get('total_pnl', 0):.2f}
Max Drawdown: {summary.get('max_drawdown', 0):.2f}%
Open Positions: {summary.get('open_positions', 0)}
        """
    
    async def shutdown(self):
        """Clean shutdown"""
        if self.session:
            await self.session.close()
        logger.info("NotificationManager shutdown complete")