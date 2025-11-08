"""
app.py - Main application to run the interview coach agent
Follows the tempApp.py pattern using InMemorySessionService and Runner
"""

import asyncio
import json
from interview_agent.agent import (
    APP_NAME,
    USER_ID,
    SESSION_ID,
    interview_runner,
    interview_agent,
    session_service,
    types
)


def format_camera_metrics(camera_data: dict) -> str:
    """Format camera metrics into readable string"""
    return f"""Camera Metrics:
- Eye Contact: {'Maintained' if camera_data.get('eye_contact_maintained') else 'Not Maintained'}
- Left Iris Position: {camera_data.get('left_iris_relative', 2.8):.3f} (valid: 2.75-2.85)
- Right Iris Position: {camera_data.get('right_iris_relative', -1.91):.3f} (valid: -1.95 to -1.875)
- Shoulder Angle: {camera_data.get('shoulder_angle', 180):.1f}° (acceptable: 165-195°)
- Head Tilt: {camera_data.get('head_tilt', 180):.1f}° (acceptable: 165-195°)
- Forward Lean: {camera_data.get('forward_lean', 0.0):.2f} (threshold: 0.15)
- Head Motion: {camera_data.get('head_motion', 0):.1f} px/frame (threshold: 15.0)
- Hand Motion: {camera_data.get('hand_motion', 0):.1f} px/frame (threshold: 20.0)"""


def format_voice_metrics(voice_data: dict) -> str:
    """Format voice metrics into readable string"""
    return f"""Voice Metrics:
- Words Per Minute: {voice_data.get('words_per_minute', 0)} (ideal: 130-160, acceptable: 120-180)
- Volume: {voice_data.get('volume_db', -50):.1f} dB (min: -60 dB)
- Clarity Score: {voice_data.get('clarity_score', 0.8):.2f} (min: 0.70)
- Transcript: "{voice_data.get('text', 'No transcript available')}" """


def create_metrics_message(camera_data: dict, voice_data: dict, question_num: int) -> str:
    """Create formatted metrics message for the agent"""
    return f"""Question {question_num} Response Analysis:

{format_camera_metrics(camera_data)}

{format_voice_metrics(voice_data)}

Please analyze these metrics and provide feedback."""


async def send_message_to_agent(message: str):
    """
    Send a message to the interview agent and print the response
    
    Args:
        message: The message or metrics data to send
    """
    print(f"\n{'='*70}")
    print(f">>> Sending to Agent:")
    print(f"{message[:200]}..." if len(message) > 200 else message)
    print(f"{'='*70}")
    
    # Create user content
    user_content = types.Content(
        role='user',
        parts=[types.Part(text=message)]
    )
    
    final_response_content = "No final response received."
    
    # Run the agent asynchronously
    async for event in interview_runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response_content = event.content.parts[0].text
    
    print(f"\n<<< Agent Response:")
    print(final_response_content)
    print(f"{'='*70}\n")
    
    # Optionally print session state
    current_session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    stored_output = current_session.state.get(interview_agent.output_key)
    
    if stored_output:
        print(f"--- Session State [{interview_agent.output_key}]:")
        try:
            print(json.dumps(json.loads(stored_output), indent=2))
        except Exception:
            print(stored_output)
        print(f"{'-'*70}\n")


async def main():
    """Main application flow"""
    print("\n" + "="*70)
    print("INTERVIEW COACH - MOCK DATA TEST")
    print("="*70)
    
    # Create session
    print("\n--- Creating Session ---")
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    print(f"✅ Session created: {SESSION_ID}")
    
    # ========================================================================
    # TEST 1: Start the interview
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 1: Starting Interview")
    print("="*70)
    
    await send_message_to_agent("Hello! I'm ready to start my practice interview.")
    
    # Wait a moment for better readability
    await asyncio.sleep(1)
    
    # ========================================================================
    # TEST 2: First question response with GOOD metrics
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 2: Question 1 Response - Good Performance")
    print("="*70)
    
    camera_data_1 = {
        "left_iris_relative": 2.80,  # Within range 2.75-2.85
        "right_iris_relative": -1.91,  # Within range -1.95 to -1.875
        "eye_contact_maintained": True,
        "shoulder_angle": 178.0,  # Within 165-195
        "head_tilt": 182.0,  # Within 165-195
        "forward_lean": 0.08,  # Below 0.15 threshold
        "head_motion": 12.5,  # Below 15.0 threshold
        "hand_motion": 18.3  # Below 20.0 threshold
    }
    
    voice_data_1 = {
        "text": "I have a background in software engineering with three years of experience at a startup.",
        "words_per_minute": 145,  # Within ideal 130-160
        "volume_db": -52.0,  # Above -60 min
        "clarity_score": 0.85  # Above 0.70 min
    }
    
    metrics_message_1 = create_metrics_message(camera_data_1, voice_data_1, 1)
    await send_message_to_agent(metrics_message_1)
    
    # Wait a moment
    await asyncio.sleep(1)
    
    # ========================================================================
    # TEST 3: Second question response with SOME ISSUES
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 3: Question 2 Response - Some Issues to Fix")
    print("="*70)
    
    camera_data_2 = {
        "left_iris_relative": 3.0,  # Outside range (too high)
        "right_iris_relative": -2.1,  # Outside range (too low)
        "eye_contact_maintained": False,
        "shoulder_angle": 172.0,  # Within range but off-center
        "head_tilt": 176.0,  # Within range but off-center
        "forward_lean": 0.18,  # Above 0.15 threshold - ISSUE
        "head_motion": 24.0,  # Above 15.0 threshold - ISSUE
        "hand_motion": 31.5  # Above 20.0 threshold - ISSUE
    }
    
    voice_data_2 = {
        "text": "My greatest strength is problem solving and working under pressure to meet deadlines.",
        "words_per_minute": 185,  # Above 180 max - ISSUE
        "volume_db": -62.0,  # Below ideal but above min
        "clarity_score": 0.76  # Slightly above min
    }
    
    metrics_message_2 = create_metrics_message(camera_data_2, voice_data_2, 2)
    await send_message_to_agent(metrics_message_2)
    
    print("\n" + "="*70)
    print("MOCK DATA TEST COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. ✅ Agent successfully processes mock data")
    print("2. 🔄 Integrate camera.py and voice.py for live data")
    print("3. 🎯 Replace mock data with real-time metrics")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())