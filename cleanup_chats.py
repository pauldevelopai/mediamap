#!/usr/bin/env python3
"""
Chat Data Cleanup Script
This script helps identify and remove bogus/test chat data from the database.
"""

import os
import sys
from datetime import datetime, timedelta
import re

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app import app, db
from backend.models import Chat, Message, User

def analyze_chats():
    """Analyze all chats to identify patterns and potential bogus data"""
    print("🔍 Analyzing chat data...")
    
    with app.app_context():
        # Get all chats
        all_chats = Chat.query.all()
        print(f"📊 Total chats found: {len(all_chats)}")
        
        # Get all messages
        all_messages = Message.query.all()
        print(f"📊 Total messages found: {len(all_messages)}")
        
        # Analyze chat patterns
        chat_analysis = {
            'empty_chats': 0,
            'test_chats': 0,
            'short_chats': 0,
            'recent_chats': 0,
            'old_chats': 0,
            'suspicious_content': 0
        }
        
        test_patterns = [
            r'test',
            r'hello',
            r'hi there',
            r'how are you',
            r'what can you do',
            r'help',
            r'example',
            r'sample',
            r'demo',
            r'debug',
            r'check',
            r'verify'
        ]
        
        suspicious_chats = []
        
        for chat in all_chats:
            messages = chat.messages
            
            # Check for empty chats
            if len(messages) == 0:
                chat_analysis['empty_chats'] += 1
                suspicious_chats.append({
                    'id': chat.id,
                    'type': 'empty',
                    'created_at': chat.created_at,
                    'message_count': 0
                })
                continue
            
            # Check for very short chats (1-2 messages)
            if len(messages) <= 2:
                chat_analysis['short_chats'] += 1
                
                # Check content for test patterns
                content = ' '.join([msg.content.lower() for msg in messages])
                is_test = any(re.search(pattern, content) for pattern in test_patterns)
                
                if is_test:
                    chat_analysis['test_chats'] += 1
                    suspicious_chats.append({
                        'id': chat.id,
                        'type': 'test_short',
                        'created_at': chat.created_at,
                        'message_count': len(messages),
                        'content': content[:100] + '...' if len(content) > 100 else content
                    })
            
            # Check for recent chats (last 24 hours)
            if chat.created_at > datetime.utcnow() - timedelta(days=1):
                chat_analysis['recent_chats'] += 1
            
            # Check for old chats (more than 30 days)
            if chat.created_at < datetime.utcnow() - timedelta(days=30):
                chat_analysis['old_chats'] += 1
        
        print("\n📈 Chat Analysis Results:")
        print(f"  Empty chats: {chat_analysis['empty_chats']}")
        print(f"  Test/short chats: {chat_analysis['test_chats']}")
        print(f"  Short chats (≤2 messages): {chat_analysis['short_chats']}")
        print(f"  Recent chats (24h): {chat_analysis['recent_chats']}")
        print(f"  Old chats (30+ days): {chat_analysis['old_chats']}")
        
        if suspicious_chats:
            print(f"\n🚨 Suspicious chats found: {len(suspicious_chats)}")
            print("\nTop 10 suspicious chats:")
            for i, chat in enumerate(suspicious_chats[:10]):
                print(f"  {i+1}. Chat ID {chat['id']} ({chat['type']}) - {chat['message_count']} messages - {chat['created_at']}")
                if 'content' in chat:
                    print(f"     Content: {chat['content']}")
        
        return suspicious_chats

def cleanup_chats(chat_ids=None, dry_run=True):
    """Clean up specified chats or use default criteria"""
    print(f"\n🧹 Starting chat cleanup (dry_run={dry_run})...")
    
    with app.app_context():
        if chat_ids:
            # Clean up specific chat IDs
            chats_to_delete = Chat.query.filter(Chat.id.in_(chat_ids)).all()
        else:
            # Use default cleanup criteria
            cutoff_date = datetime.utcnow() - timedelta(days=7)  # Keep chats from last 7 days
            
            # Find chats to delete based on criteria
            chats_to_delete = []
            
            # 1. Empty chats
            empty_chats = Chat.query.outerjoin(Message).filter(Message.id.is_(None)).all()
            chats_to_delete.extend(empty_chats)
            
            # 2. Very old chats (more than 30 days)
            old_chats = Chat.query.filter(Chat.created_at < datetime.utcnow() - timedelta(days=30)).all()
            chats_to_delete.extend(old_chats)
            
            # 3. Test chats with specific patterns
            test_patterns = [r'test', r'hello', r'hi there', r'how are you', r'what can you do']
            all_chats = Chat.query.all()
            
            for chat in all_chats:
                if chat in chats_to_delete:  # Skip if already marked for deletion
                    continue
                    
                messages = chat.messages
                if len(messages) <= 2:  # Short chats
                    content = ' '.join([msg.content.lower() for msg in messages])
                    if any(re.search(pattern, content) for pattern in test_patterns):
                        chats_to_delete.append(chat)
        
        # Remove duplicates
        chats_to_delete = list(set(chats_to_delete))
        
        print(f"📋 Chats to delete: {len(chats_to_delete)}")
        
        if not dry_run:
            # Actually delete the chats
            deleted_count = 0
            for chat in chats_to_delete:
                try:
                    print(f"🗑️ Deleting chat {chat.id} ({len(chat.messages)} messages)")
                    db.session.delete(chat)
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ Error deleting chat {chat.id}: {e}")
            
            db.session.commit()
            print(f"✅ Successfully deleted {deleted_count} chats")
        else:
            # Just show what would be deleted
            print("\n📋 Chats that would be deleted:")
            for i, chat in enumerate(chats_to_delete[:20]):  # Show first 20
                print(f"  {i+1}. Chat ID {chat.id} - {len(chat.messages)} messages - {chat.created_at}")
            
            if len(chats_to_delete) > 20:
                print(f"  ... and {len(chats_to_delete) - 20} more")
        
        return len(chats_to_delete)

def interactive_cleanup():
    """Interactive cleanup with user confirmation"""
    print("🎯 Interactive Chat Cleanup")
    print("=" * 50)
    
    # First analyze
    suspicious_chats = analyze_chats()
    
    if not suspicious_chats:
        print("✅ No suspicious chats found!")
        return
    
    print(f"\n🚨 Found {len(suspicious_chats)} suspicious chats")
    
    # Ask user what to do
    print("\nOptions:")
    print("1. Delete all suspicious chats")
    print("2. Delete only empty chats")
    print("3. Delete only test chats")
    print("4. Delete old chats (30+ days)")
    print("5. Custom cleanup")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == '1':
        chat_ids = [chat['id'] for chat in suspicious_chats]
        confirm = input(f"Delete {len(chat_ids)} suspicious chats? (y/N): ").strip().lower()
        if confirm == 'y':
            cleanup_chats(chat_ids, dry_run=False)
        else:
            print("❌ Cancelled")
    
    elif choice == '2':
        empty_chat_ids = [chat['id'] for chat in suspicious_chats if chat['type'] == 'empty']
        confirm = input(f"Delete {len(empty_chat_ids)} empty chats? (y/N): ").strip().lower()
        if confirm == 'y':
            cleanup_chats(empty_chat_ids, dry_run=False)
        else:
            print("❌ Cancelled")
    
    elif choice == '3':
        test_chat_ids = [chat['id'] for chat in suspicious_chats if chat['type'] == 'test_short']
        confirm = input(f"Delete {len(test_chat_ids)} test chats? (y/N): ").strip().lower()
        if confirm == 'y':
            cleanup_chats(test_chat_ids, dry_run=False)
        else:
            print("❌ Cancelled")
    
    elif choice == '4':
        # Delete old chats
        with app.app_context():
            old_chats = Chat.query.filter(Chat.created_at < datetime.utcnow() - timedelta(days=30)).all()
            old_chat_ids = [chat.id for chat in old_chats]
            confirm = input(f"Delete {len(old_chat_ids)} old chats (30+ days)? (y/N): ").strip().lower()
            if confirm == 'y':
                cleanup_chats(old_chat_ids, dry_run=False)
            else:
                print("❌ Cancelled")
    
    elif choice == '5':
        print("\nCustom cleanup options:")
        print("1. Delete chats older than X days")
        print("2. Delete chats with specific content")
        print("3. Delete chats from specific user")
        
        custom_choice = input("Enter choice (1-3): ").strip()
        
        if custom_choice == '1':
            days = input("Delete chats older than how many days? ").strip()
            try:
                days = int(days)
                with app.app_context():
                    old_chats = Chat.query.filter(Chat.created_at < datetime.utcnow() - timedelta(days=days)).all()
                    old_chat_ids = [chat.id for chat in old_chats]
                    confirm = input(f"Delete {len(old_chat_ids)} chats older than {days} days? (y/N): ").strip().lower()
                    if confirm == 'y':
                        cleanup_chats(old_chat_ids, dry_run=False)
                    else:
                        print("❌ Cancelled")
            except ValueError:
                print("❌ Invalid number")
    
    elif choice == '6':
        print("👋 Exiting")
        return
    
    else:
        print("❌ Invalid choice")

def main():
    """Main function"""
    print("🧹 Chat Data Cleanup Tool")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'analyze':
            analyze_chats()
        elif command == 'cleanup':
            dry_run = '--dry-run' in sys.argv
            cleanup_chats(dry_run=dry_run)
        elif command == 'interactive':
            interactive_cleanup()
        else:
            print("❌ Unknown command. Use: analyze, cleanup, or interactive")
    else:
        # Default to interactive mode
        interactive_cleanup()

if __name__ == "__main__":
    main()







