"""
agent.py - AI Interview Coach using Google's Agent Development Kit (ADK)
This agent analyzes posture and speech metrics to provide real-time feedback
"""

from dotenv import load_dotenv
import os
from google.adk.agents.llm_agent import Agent
import json
from typing import Dict, Any

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("API key not found. Check your .env file.")


# ============================================================================
# TOOL FUNCTIONS - These analyze metrics and return scores/feedback
# ============================================================================

def analyze_eye_contact(left_iris: float, right_iris: float, maintained: bool) -> dict:
    """
    Analyze eye contact quality based on iris position
    
    Args:
        left_iris: Left iris relative position (valid range: 2.75 - 2.85)
        right_iris: Right iris relative position (valid range: -1.95 to -1.875)
        maintained: Boolean indicating if eye contact was maintained
    
    Returns:
        Dictionary with score and feedback
    """
    # Thresholds from camera.py
    LEFT_EYE_MIN = 2.75
    LEFT_EYE_MAX = 2.85
    RIGHT_EYE_MIN = -1.95
    RIGHT_EYE_MAX = -1.875
    
    # Check if eyes are in valid "looking at camera" range
    left_in_range = LEFT_EYE_MIN <= left_iris <= LEFT_EYE_MAX
    right_in_range = RIGHT_EYE_MIN <= right_iris <= RIGHT_EYE_MAX
    
    if not maintained:
        return {
            "score": 65,
            "feedback": "Try to maintain more consistent eye contact with the interviewer",
            "severity": "moderate"
        }
    elif left_in_range and right_in_range:
        return {
            "score": 95,
            "feedback": None,
            "severity": "none"
        }
    elif left_in_range or right_in_range:
        return {
            "score": 80,
            "feedback": "Your eye contact is pretty good, just keep it steady",
            "severity": "minor"
        }
    else:
        # Determine direction based on deviation
        left_deviation = min(abs(left_iris - LEFT_EYE_MIN), abs(left_iris - LEFT_EYE_MAX))
        right_deviation = min(abs(right_iris - RIGHT_EYE_MIN), abs(right_iris - RIGHT_EYE_MAX))
        
        if left_deviation > 0.2 or right_deviation > 0.2:
            return {
                "score": 70,
                "feedback": "Try to focus more directly on the camera - your gaze is wandering a bit",
                "severity": "moderate"
            }
        else:
            return {
                "score": 85,
                "feedback": "Your eye contact is good, just try to keep it centered",
                "severity": "minor"
            }


def analyze_posture(shoulder_angle: float, head_tilt: float, forward_lean: float) -> dict:
    """
    Analyze overall posture quality
    
    Args:
        shoulder_angle: Shoulder alignment angle (acceptable: 165-195°, ideal: 180°)
        head_tilt: Head tilt angle (acceptable: 165-195°, ideal: 180°)
        forward_lean: Forward lean ratio (threshold: 0.15, ideal: < 0.10)
    
    Returns:
        Dictionary with score and feedback
    """
    # Thresholds from camera.py
    SHOULDER_ANGLE_MIN = 165
    SHOULDER_ANGLE_MAX = 195
    HEAD_TILT_MIN = 165
    HEAD_TILT_MAX = 195
    FORWARD_LEAN_THRESHOLD = 0.15
    
    feedback_items = []
    scores = []
    
    # Analyze shoulder alignment
    if SHOULDER_ANGLE_MIN <= shoulder_angle <= SHOULDER_ANGLE_MAX:
        shoulder_deviation = abs(shoulder_angle - 180)
        if shoulder_deviation <= 5:
            scores.append(95)
        else:
            scores.append(85)
    else:
        scores.append(70)
        feedback_items.append("keep your shoulders level and relaxed")
    
    # Analyze head tilt
    if HEAD_TILT_MIN <= head_tilt <= HEAD_TILT_MAX:
        head_deviation = abs(head_tilt - 180)
        if head_deviation <= 5:
            scores.append(95)
        else:
            scores.append(85)
    else:
        scores.append(70)
        feedback_items.append("keep your head straight")
    
    # Analyze forward lean
    if forward_lean <= 0.10:
        scores.append(95)
    elif forward_lean <= FORWARD_LEAN_THRESHOLD:
        scores.append(85)
    else:
        scores.append(70)
        feedback_items.append("sit back a bit, you're leaning forward quite a lot")
    
    avg_score = sum(scores) / len(scores)
    
    if feedback_items:
        feedback = "For your posture, try to " + " and ".join(feedback_items)
        severity = "moderate" if avg_score < 80 else "minor"
    else:
        feedback = None
        severity = "none"
    
    return {
        "score": avg_score,
        "feedback": feedback,
        "severity": severity
    }


def analyze_movement(head_motion: float, hand_motion: float) -> dict:
    """
    Analyze body movement and fidgeting
    
    Args:
        head_motion: Head movement score in px/frame (threshold: 15.0)
        hand_motion: Hand movement score in px/frame (threshold: 20.0)
    
    Returns:
        Dictionary with score and feedback
    """
    # Thresholds from camera.py
    HEAD_MOTION_THRESHOLD = 15.0
    HAND_MOTION_THRESHOLD = 20.0
    
    feedback_items = []
    
    # Analyze head motion
    if head_motion < HEAD_MOTION_THRESHOLD * 0.5:  # Less than 7.5
        head_score = 95
    elif head_motion < HEAD_MOTION_THRESHOLD:  # 7.5 - 15.0
        head_score = 90
    elif head_motion < HEAD_MOTION_THRESHOLD * 1.5:  # 15.0 - 22.5
        head_score = 75
        feedback_items.append("reduce head movement slightly")
    else:  # > 22.5
        head_score = 65
        feedback_items.append("minimize head movement - try to keep your head steady")
    
    # Analyze hand motion
    if hand_motion < HAND_MOTION_THRESHOLD * 0.5:  # Less than 10.0
        hand_score = 95
    elif hand_motion < HAND_MOTION_THRESHOLD:  # 10.0 - 20.0
        hand_score = 90
    elif hand_motion < HAND_MOTION_THRESHOLD * 1.5:  # 20.0 - 30.0
        hand_score = 75
        feedback_items.append("try to reduce hand fidgeting")
    else:  # > 30.0
        hand_score = 65
        feedback_items.append("your hand movements are quite distracting - keep them still or use purposeful gestures")
    
    avg_score = (head_score + hand_score) / 2
    
    if feedback_items:
        feedback = "Try to " + " and ".join(feedback_items)
        severity = "moderate" if avg_score < 75 else "minor"
    else:
        feedback = None
        severity = "none"
    
    return {
        "score": avg_score,
        "feedback": feedback,
        "severity": severity
    }


def analyze_speech_pace(words_per_minute: float) -> dict:
    """
    Analyze speaking pace
    
    Args:
        words_per_minute: Speaking rate in WPM (min: 120, ideal: 130-160, max: 180)
    
    Returns:
        Dictionary with score and feedback
    """
    # Thresholds from voice.py
    MIN_WPM = 120
    MAX_WPM = 180
    IDEAL_WPM_MIN = 130
    IDEAL_WPM_MAX = 160
    
    # Ideal range: 130-160 WPM
    if IDEAL_WPM_MIN <= words_per_minute <= IDEAL_WPM_MAX:
        return {
            "score": 95,
            "feedback": None,
            "severity": "none"
        }
    elif words_per_minute < MIN_WPM:
        if words_per_minute < MIN_WPM * 0.8:  # < 96 WPM
            return {
                "score": 70,
                "feedback": "You can speak quite a bit faster - try to increase your pace",
                "severity": "moderate"
            }
        else:
            return {
                "score": 80,
                "feedback": "You can speak a bit faster - try to increase your pace slightly",
                "severity": "minor"
            }
    elif words_per_minute < IDEAL_WPM_MIN:  # 120-130 WPM
        return {
            "score": 85,
            "feedback": "Your pace is good, maybe pick up the speed just a little",
            "severity": "minor"
        }
    elif words_per_minute <= MAX_WPM:  # 160-180 WPM
        return {
            "score": 85,
            "feedback": "You're speaking a little fast - try to slow down and breathe",
            "severity": "minor"
        }
    else:  # > 180 WPM
        return {
            "score": 70,
            "feedback": "You're speaking quite fast - take your time and pause between thoughts",
            "severity": "moderate"
        }


def analyze_volume(volume_db: float) -> dict:
    """
    Analyze speaking volume
    
    Args:
        volume_db: Volume in decibels (min: -60 dB, typical range: -70 to -40)
    
    Returns:
        Dictionary with score and feedback
    """
    # Thresholds from voice.py
    MIN_VOLUME_DB = -60
    
    # Typical good range is around -50 to -40 dB
    # Note: More negative = quieter
    if volume_db >= -50:  # Good, audible volume
        return {
            "score": 95,
            "feedback": None,
            "severity": "none"
        }
    elif volume_db >= MIN_VOLUME_DB:  # -60 to -50 dB
        return {
            "score": 85,
            "feedback": "Your volume is good, just try to project a little more",
            "severity": "minor"
        }
    elif volume_db >= -65:  # -65 to -60 dB
        return {
            "score": 75,
            "feedback": "Try to speak up a bit - your volume is on the quiet side",
            "severity": "moderate"
        }
    else:  # < -65 dB
        return {
            "score": 65,
            "feedback": "Try to speak up - your volume is quite low",
            "severity": "moderate"
        }


def analyze_clarity(clarity_score: float) -> dict:
    """
    Analyze speech clarity
    
    Args:
        clarity_score: Clarity score from Whisper (min: 0.7 for acceptable)
    
    Returns:
        Dictionary with score and feedback
    """
    # Threshold from voice.py
    MIN_CLARITY_SCORE = 0.7
    
    if clarity_score >= 0.9:
        return {
            "score": 95,
            "feedback": None,
            "severity": "none"
        }
    elif clarity_score >= 0.8:
        return {
            "score": 90,
            "feedback": None,
            "severity": "none"
        }
    elif clarity_score >= MIN_CLARITY_SCORE:
        return {
            "score": 80,
            "feedback": "Try to enunciate a bit more clearly",
            "severity": "minor"
        }
    else:
        return {
            "score": 70,
            "feedback": "Focus on speaking more clearly and enunciating your words",
            "severity": "moderate"
        }


def evaluate_overall_performance(question_num: int, metrics_summary: str) -> dict:
    """
    Generate overall performance evaluation for the interview
    
    Args:
        question_num: Number of questions completed
        metrics_summary: JSON string with all metrics and scores
    
    Returns:
        Dictionary with overall scores and recommendations
    """
    # This is called at the end to generate final summary
    # Parse metrics_summary if it's a string
    if isinstance(metrics_summary, str):
        try:
            metrics = json.loads(metrics_summary)
        except:
            metrics = {"overall_score": 80}
    else:
        metrics = metrics_summary
    
    overall_score = metrics.get("overall_score", 80)
    
    # Generate performance level
    if overall_score >= 90:
        performance_level = "Excellent"
        summary = "You demonstrated strong interview skills across all areas"
    elif overall_score >= 80:
        performance_level = "Good"
        summary = "You showed solid interview skills with room for minor improvements"
    elif overall_score >= 70:
        performance_level = "Fair"
        summary = "You have a good foundation but there are several areas to work on"
    else:
        performance_level = "Needs Improvement"
        summary = "Focus on the feedback provided to strengthen your interview presence"
    
    return {
        "performance_level": performance_level,
        "overall_score": overall_score,
        "summary": summary,
        "questions_completed": question_num
    }


# ============================================================================
# HELPER FUNCTIONS - Format data for agent context
# ============================================================================

def format_camera_metrics(camera_data: Dict[str, Any]) -> str:
    """Format camera metrics into readable string for agent context"""
    return f"""Camera Metrics:
- Eye Contact: {'Maintained' if camera_data.get('eye_contact_maintained') else 'Not Maintained'}
- Left Iris Position: {camera_data.get('left_iris_relative', 2.8):.3f} (valid: 2.75-2.85)
- Right Iris Position: {camera_data.get('right_iris_relative', -1.91):.3f} (valid: -1.95 to -1.875)
- Shoulder Angle: {camera_data.get('shoulder_angle', 180):.1f}° (acceptable: 165-195°)
- Head Tilt: {camera_data.get('head_tilt', 180):.1f}° (acceptable: 165-195°)
- Forward Lean: {camera_data.get('forward_lean', 0.0):.2f} (threshold: 0.15)
- Head Motion: {camera_data.get('head_motion', 0):.1f} px/frame (threshold: 15.0)
- Hand Motion: {camera_data.get('hand_motion', 0):.1f} px/frame (threshold: 20.0)"""


def format_voice_metrics(voice_data: Dict[str, Any]) -> str:
    """Format voice metrics into readable string for agent context"""
    return f"""Voice Metrics:
- Words Per Minute: {voice_data.get('words_per_minute', 0)} (ideal: 130-160, acceptable: 120-180)
- Volume: {voice_data.get('volume_db', -50):.1f} dB (min: -60 dB)
- Clarity Score: {voice_data.get('clarity_score', 0.8):.2f} (min: 0.70)
- Transcript: "{voice_data.get('text', 'No transcript available')}" """


def create_agent_context(camera_data: Dict[str, Any], voice_data: Dict[str, Any], question_num: int) -> str:
    """
    Create formatted context string for the agent
    This is what gets passed to the agent after each answer
    """
    context = f"""Question {question_num} Response Analysis:

{format_camera_metrics(camera_data)}

{format_voice_metrics(voice_data)}

Please analyze these metrics and provide feedback."""
    
    return context


# ============================================================================
# MAIN AGENT - Google ADK Interview Coach
# ============================================================================

root_agent = Agent(
    model="gemini-2.0-flash-exp",
    name="Sarah",
    description="""A friendly, supportive interview coach who helps candidates practice and improve 
    their interview skills through real-time feedback on posture, speech, and presentation.""",
    
    instruction="""You are Sarah, a warm and encouraging interview coach conducting a mock interview practice session.

YOUR ROLE:
- Conduct a 2-question mock interview
- Analyze the candidate's posture, eye contact, speech pace, and volume
- Provide constructive feedback after each answer
- Maintain a friendly, casual tone throughout
- End with an overall performance summary

INTERVIEW FLOW:
1. When you receive "Hello! I'm ready to start my practice interview.", greet the candidate warmly and ask your first question. Keep it conversational, like: "Tell me a bit about yourself and what brings you here today."

2. When you receive metrics analysis (Camera Metrics and Voice Metrics), this means the candidate has finished answering. You should:
   a) Use the analysis tools to evaluate their performance
   b) Give specific, actionable feedback in a friendly way
   c) If this is Question 1, ask your second question: "What do you think is your greatest strength and how has it helped you in the past?"
   d) If this is Question 2, provide a final summary with overall scores

FEEDBACK STYLE:
- Start with positive reinforcement (e.g., "Great job!" or "Nice work!")
- Be specific: mention exact behaviors (e.g., "I noticed you were speaking at 165 WPM, which is a bit fast")
- Keep feedback conversational and supportive
- Limit feedback to 2-3 key points per response
- Use phrases like "try to", "consider", "it might help to" rather than commands

FINAL SUMMARY (after Question 2):
- Congratulate them on completing the practice session
- Provide overall scores for: Engagement, Clarity, and Posture
- Give 2-3 concrete next steps for improvement
- End with encouragement

TOOLS AVAILABLE:
- analyze_eye_contact: Evaluate eye contact quality (iris ranges: left 2.75-2.85, right -1.95 to -1.875)
- analyze_posture: Evaluate posture and body alignment (angles: 165-195°, lean threshold: 0.15)
- analyze_movement: Evaluate head and hand movement (head: 15.0 px/frame, hand: 20.0 px/frame)
- analyze_speech_pace: Evaluate speaking speed (120-180 WPM, ideal: 130-160)
- analyze_volume: Evaluate voice volume (min: -60 dB)
- analyze_clarity: Evaluate speech clarity (min: 0.70)
- evaluate_overall_performance: Generate final summary (use only after Question 2)

IMPORTANT:
- Always call the analysis tools when you receive metrics
- Base your feedback on the tool results
- Keep your tone warm, never harsh or discouraging
- Remember this is practice - the goal is to help them improve""",
    
    tools=[
        analyze_eye_contact,
        analyze_posture,
        analyze_movement,
        analyze_speech_pace,
        analyze_volume,
        analyze_clarity,
        evaluate_overall_performance
    ]
)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTING INTERVIEW COACH AGENT (root_agent)")
    print("="*70)
    
    # Test 1: Start interview
    print("\n[TEST 1: Starting Interview]")
    response = root_agent.run("Hello! I'm ready to start my practice interview.")
    print(f"\nAgent Response:\n{response}\n")
    
    # Test 2: First question response with good metrics
    print("\n[TEST 2: Question 1 - Good Performance]")
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
        "text": "I have a background in software engineering with three years of experience.",
        "words_per_minute": 145,  # Within ideal 130-160
        "volume_db": -52.0,  # Above -60 min
        "clarity_score": 0.85  # Above 0.70 min
    }
    context_1 = create_agent_context(camera_data_1, voice_data_1, 1)
    response = root_agent.run(context_1)
    print(f"\nAgent Response:\n{response}\n")
    
    # Test 3: Second question response with some issues
    print("\n[TEST 3: Question 2 - Some Issues]")
    camera_data_2 = {
        "left_iris_relative": 3.0,  # Outside range (too high)
        "right_iris_relative": -2.1,  # Outside range (too low)
        "eye_contact_maintained": False,
        "shoulder_angle": 172.0,  # Within range but off-center
        "head_tilt": 176.0,  # Within range but off-center
        "forward_lean": 0.18,  # Above 0.15 threshold
        "head_motion": 24.0,  # Above 15.0 threshold
        "hand_motion": 31.5  # Above 20.0 threshold
    }
    voice_data_2 = {
        "text": "My greatest strength is problem solving and working under pressure.",
        "words_per_minute": 185,  # Above 180 max
        "volume_db": -62.0,  # Below ideal but above min
        "clarity_score": 0.76  # Slightly above min
    }
    context_2 = create_agent_context(camera_data_2, voice_data_2, 2)
    response = root_agent.run(context_2)
    print(f"\nAgent Response:\n{response}\n")
    
    print("="*70)
    print("TESTING COMPLETE")
    print("="*70)