"""
Utility functions for context engineering notebooks.
"""

from rich.console import Console
from rich.panel import Panel
from rich.markup import escape

import json

console = Console()

def format_message_content(message):
    """Convert message content to displayable string"""
    
    # Check if this is an AI message with tool calls
    if hasattr(message, 'tool_calls') and message.tool_calls:
        parts = []
        
        # Add regular content if it exists and is not empty
        if isinstance(message.content, str) and message.content.strip():
            parts.append(message.content)
        elif isinstance(message.content, list):
            for item in message.content:
                if item.get('type') == 'text' and item.get('text', '').strip():
                    parts.append(item['text'])
        
        # Add tool calls information
        for tool_call in message.tool_calls:
            parts.append(f"\n🔧 Tool Call: {tool_call['name']}")
            if 'args' in tool_call:
                parts.append(f"   Args: {json.dumps(tool_call['args'], indent=2)}")
            elif 'function' in tool_call and 'arguments' in tool_call['function']:
                # Handle the case where arguments are in function.arguments
                try:
                    args_dict = json.loads(tool_call['function']['arguments'])
                    parts.append(f"   Args: {json.dumps(args_dict, indent=2)}")
                except json.JSONDecodeError:
                    parts.append(f"   Args: {tool_call['function']['arguments']}")
            
            # Add tool call ID if available
            if 'id' in tool_call:
                parts.append(f"   ID: {tool_call['id']}")
        
        return "\n".join(parts) if parts else "Tool call with no displayable content"
    
    # Handle regular content
    elif isinstance(message.content, str):
        return message.content
    elif isinstance(message.content, list):
        # Handle complex content like tool calls in content
        parts = []
        for item in message.content:
            if item.get('type') == 'text':
                parts.append(item['text'])
            elif item.get('type') == 'tool_use':
                parts.append(f"\n🔧 Tool Call: {item['name']}")
                parts.append(f"   Args: {json.dumps(item['input'], indent=2)}")
        return "\n".join(parts)
    else:
        return str(message.content)

def format_message(message):
    """Format and display a message with proper styling"""
    
    msg_type = message.__class__.__name__.replace('Message', '')
    is_tool_call_req = False
    
    if msg_type == 'AI':
        is_tool_call_req = bool(getattr(message, 'tool_calls', None))
    
    content = format_message_content(message)
    content = escape(str(content))

    if msg_type == 'Human':
        console.print(Panel(content, title="🧑 Human", border_style="blue"))
    elif msg_type == 'AI' and is_tool_call_req:
        console.print(Panel(content, title="⚙️ Tool Request", border_style="red"))
    elif msg_type == 'AI' and not is_tool_call_req:
        console.print(Panel(content, title="🤖 Assistant", border_style="green"))
    elif msg_type == 'Tool':
        console.print(Panel(content, title="🔧 Tool Output", border_style="yellow"))
    else:
        console.print(Panel(content, title=f"📝 {msg_type}", border_style="white"))



def format_streaming_results_simple(data_part):
    try:
        data = json.loads(data_part)
        
        if data.get('type') == 'tool_start':
            tool = data.get('tool', '').replace('_', ' ').title()
            print(f"\n🔧 {tool}...")
            
        elif data.get('type') == 'tool_end':
            print("   ✅ Done\n")
            
        elif data.get('type') == 'token':
            content = data.get('content', '')
            if '- Final Decision:' in content:
                print("📋 Final Assessment:")
                print("─" * 30)
            print(content, end='', flush=True)
            
    except json.JSONDecodeError:
        pass


def format_streaming_results_detailed(data_part: str, state: dict) -> bool:
    """
    Pretty-print one SSE 'data:' payload and update state.
    """
    try:
        data = json.loads(data_part)

        # tool_start
        if data.get('type') == 'tool_start':
            state['tool_counter'] = state.get('tool_counter', 0) + 1
            tool_name = data.get('tool', 'Unknown')
            print(f"\n{'─' * 60}")
            print(f"🔧 STEP {state['tool_counter']}: {tool_name.upper()}")
            print(f"{'─' * 60}")
            print(f"⏳ {data.get('message', 'Processing...')}")
            state['current_section'] = tool_name

        # tool_end
        elif data.get('type') == 'tool_end':
            tool_name = data.get('tool', state.get('current_section'))
            print(f"\n✅ COMPLETED: {tool_name}")
            if 'summary' in data:
                summary = data['summary']
                if len(summary) > 150:
                    summary = summary[:150] + "..."
                print(f"📄 Summary: {summary}")
            print(f"{'─' * 60}\n")

        # token stream
        elif data.get('type') == 'token':
            content = data.get('content', '')
            if content.strip().startswith('- Final Decision:'):
                print(f"\n{'═' * 60}")
                print("🏁 FINAL MEDICAL REVIEW DECISION")
                print(f"{'═' * 60}")
            print(content, end='', flush=True)

        # analysis complete
        elif data.get('type') == 'complete':
            print(f"\n\n✨ {data.get('message', 'Analysis complete')}")

        # status updates
        elif 'status' in data:
            if 'starting' in data.get('status', '').lower():
                print(f"🚀 {data.get('message', '')}")
            else:
                print(f"📡 {data.get('message', '')}")

        # error
        elif 'error' in data:
            print(f"\n❌ ERROR: {data['error']}")
            print("=" * 80)

        # unknown payloads -> just ignore

    except json.JSONDecodeError:
        # Non-JSON data in the SSE stream (e.g., logs or plain text)
        if data_part.strip():
            print(f"📝 {data_part}")

