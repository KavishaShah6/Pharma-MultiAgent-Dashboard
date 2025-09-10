from datetime import datetime
import json

class MessageBroker:
    """Simple message broker for agent communication"""
    
    def __init__(self):
        self.messages = []
        self.subscribers = {}
        
    def publish(self, topic, message, sender):
        """Publish a message to a topic"""
        msg = {
            'topic': topic,
            'message': message,
            'sender': sender,
            'timestamp': datetime.now().isoformat()
        }
        
        self.messages.append(msg)
        
        # Notify subscribers
        if topic in self.subscribers:
            for subscriber in self.subscribers[topic]:
                subscriber(msg)
                
    def subscribe(self, topic, callback):
        """Subscribe to a topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        
    def get_messages(self, topic=None, since=None):
        """Get messages for a topic"""
        messages = self.messages
        
        if topic:
            messages = [m for m in messages if m['topic'] == topic]
            
        if since:
            messages = [m for m in messages if m['timestamp'] > since]
            
        return messages
    
    def clear_messages(self):
        """Clear all messages"""
        self.messages = []