import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import argparse
import numpy as np
from collections import Counter

def load_log_data(log_file="D:\\real-time-cam-v1\\behavior_logs\\behavior_log_20250412_072936.csv"):
    """Load and parse the behavior log CSV file"""
    try:
        df = pd.read_csv(log_file)
        print(f"Successfully loaded {len(df)} records from {log_file}")
        return df
    except Exception as e:
        print(f"Error loading log file: {e}")
        return None

def preprocess_data(df):
    """Preprocess the dataframe for analysis"""
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by name and timestamp
    df = df.sort_values(['name', 'timestamp'])
    
    # Extract attention level from attention_state
    df['attention_level'] = df['attention_state'].apply(
        lambda x: 'High' if 'FOCUSED' in str(x) 
                 else ('Medium' if 'TALKING' in str(x) or 'TIRED' in str(x) or 'CONCERNED' in str(x)
                       else 'Low' if 'DISTRACTED' in str(x) or 'AWAY' in str(x) or 'SLEEPING' in str(x)
                       else 'Unknown')
    )
    
    return df

def get_student_names(df):
    """Get unique student names from the dataframe"""
    return df['name'].unique()

def calculate_attention_percentages(student_data):
    """Calculate attention level percentages for a student"""
    attention_counts = student_data['attention_level'].value_counts()
    total = len(student_data)
    
    high = attention_counts.get('High', 0) / total * 100 if total > 0 else 0
    medium = attention_counts.get('Medium', 0) / total * 100 if total > 0 else 0
    low = attention_counts.get('Low', 0) / total * 100 if total > 0 else 0
    unknown = attention_counts.get('Unknown', 0) / total * 100 if total > 0 else 0
    
    return {
        'High': high,
        'Medium': medium,
        'Low': low,
        'Unknown': unknown
    }

def generate_attention_chart(student_data, output_path):
    """Generate an attention distribution chart"""
    attention_percentages = calculate_attention_percentages(student_data)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create pie chart
    labels = [f"{k} ({v:.1f}%)" for k, v in attention_percentages.items() if v > 0]
    sizes = [v for v in attention_percentages.values() if v > 0]
    colors = ['#4CAF50', '#FFC107', '#F44336', '#9E9E9E']
    explode = [0.1 if label.startswith('High') else 0 for label in labels]
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.title('Attention Level Distribution')
    
    # Save the chart
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def analyze_distractions(student_data):
    """Analyze the main sources of distraction"""
    # Filter to only distracted states
    distracted = student_data[student_data['attention_state'].str.contains('DISTRACTED', na=False)]
    
    if len(distracted) == 0:
        return "No distractions detected."
    
    # Count distraction reasons
    reasons = []
    for state in distracted['attention_state']:
        if ':' in state:
            reason = state.split(':')[1].strip()
            reasons.append(reason)
    
    # Get the counts
    reason_counts = Counter(reasons)
    total = sum(reason_counts.values())
    
    # Format the result
    result = "Main sources of distraction:\n"
    for reason, count in reason_counts.most_common():
        percentage = (count / total) * 100
        result += f"- {reason}: {count} instances ({percentage:.1f}%)\n"
    
    return result

def analyze_posture(student_data):
    """Analyze posture data"""
    posture_counts = student_data['posture_status'].value_counts()
    total = len(student_data)
    
    if total == 0:
        return "No posture data available."
    
    good_count = posture_counts.get('Good Posture', 0)
    bad_count = posture_counts.get('Bad Posture', 0)
    unknown_count = posture_counts.get('Unknown', 0) + posture_counts.get('Posture error', 0)
    
    good_pct = (good_count / total) * 100
    bad_pct = (bad_count / total) * 100
    
    if good_count > bad_count:
        assessment = f"Overall good posture, with proper positioning {good_pct:.1f}% of the time."
    elif bad_count > good_count:
        assessment = f"Posture needs improvement, with poor positioning {bad_pct:.1f}% of the time."
    else:
        assessment = "Posture data is inconclusive."
    
    return assessment

def analyze_emotion(student_data):
    """Analyze emotion data"""
    emotions = student_data['emotion'].value_counts()
    total = sum(emotions)
    
    if total == 0:
        return "No emotion data available."
    
    # Group emotions
    positive = sum(emotions.get(e, 0) for e in ['happy', 'surprise'])
    negative = sum(emotions.get(e, 0) for e in ['sad', 'angry', 'fear', 'disgust'])
    neutral = emotions.get('neutral', 0)
    unknown = emotions.get('Unknown', 0)
    
    # Calculate percentages
    positive_pct = (positive / total) * 100 if total > 0 else 0
    negative_pct = (negative / total) * 100 if total > 0 else 0
    neutral_pct = (neutral / total) * 100 if total > 0 else 0
    
    # Determine dominant emotion
    dominant_emotion = emotions.idxmax() if not emotions.empty else "Unknown"
    dominant_count = emotions.max() if not emotions.empty else 0
    dominant_pct = (dominant_count / total) * 100 if total > 0 else 0
    
    if positive > negative and positive > neutral:
        assessment = f"Generally positive emotional state ({positive_pct:.1f}% of observations)."
    elif negative > positive and negative > neutral:
        assessment = f"Shows signs of negative emotions ({negative_pct:.1f}% of observations)."
    elif neutral > positive and neutral > negative:
        assessment = f"Mostly neutral emotional expression ({neutral_pct:.1f}% of observations)."
    else:
        assessment = "Mixed emotional states with no clear pattern."
    
    assessment += f" Dominant emotion: {dominant_emotion} ({dominant_pct:.1f}%)."
    
    return assessment

def analyze_gaze(student_data):
    """Analyze gaze tracking data"""
    gaze_counts = student_data['gaze_status'].value_counts()
    total = len(student_data)
    
    if total == 0:
        return "No gaze tracking data available."
    
    center_count = gaze_counts.get('Looking center', 0)
    looking_away = gaze_counts.get('Looking left', 0) + gaze_counts.get('Looking right', 0)
    blinking = gaze_counts.get('Blinking', 0)
    undetected = gaze_counts.get('Gaze undetected', 0) + gaze_counts.get('Gaze error', 0)
    
    center_pct = (center_count / total) * 100 if total > 0 else 0
    away_pct = (looking_away / total) * 100 if total > 0 else 0
    
    if center_count > looking_away:
        assessment = f"Maintained focus well, looking at center {center_pct:.1f}% of the time."
    elif looking_away > center_count:
        assessment = f"Frequently looking away from center ({away_pct:.1f}% of observations)."
    else:
        assessment = "Mixed gaze patterns with no clear trend."
    
    return assessment

def analyze_behaviors(student_data):
    """Analyze specific behaviors like talking, sleeping, etc."""
    behaviors = student_data['behavior'].value_counts()
    
    if len(behaviors) == 0:
        return "No specific behaviors detected."
    
    result = "Behavior analysis:\n"
    total = len(student_data)
    
    # Check for sleeping
    sleeping = behaviors.get('Sleeping', 0)
    sleeping_pct = (sleeping / total) * 100 if total > 0 else 0
    if sleeping > 0:
        result += f"- Appeared to be sleeping during {sleeping_pct:.1f}% of observations.\n"
    
    # Check for talking
    talking = behaviors.get('Talking', 0)
    talking_pct = (talking / total) * 100 if total > 0 else 0
    if talking > 0:
        result += f"- Was talking during {talking_pct:.1f}% of observations.\n"
    
    # Check for phone use
    phone = behaviors.get('Using Phone', 0)
    phone_pct = (phone / total) * 100 if total > 0 else 0
    if phone > 0:
        result += f"- Was using phone during {phone_pct:.1f}% of observations.\n"
    
    # Check attentiveness
    attentive = behaviors.get('Attentive', 0)
    attentive_pct = (attentive / total) * 100 if total > 0 else 0
    if attentive > 0:
        result += f"- Was attentive during {attentive_pct:.1f}% of observations.\n"
    
    return result

def generate_recommendations(student_data):
    """Generate personalized recommendations based on the analysis"""
    recommendations = []
    
    # Analyze attention levels
    attention_percentages = calculate_attention_percentages(student_data)
    if attention_percentages['Low'] > 30:
        recommendations.append("Work on maintaining focus during class. Consider removing distractions like phones.")
    
    # Check for phone distractions
    phone_detected = student_data['phone_detected'].sum()
    if phone_detected > 0:
        recommendations.append("Keep phone away during class to minimize distractions.")
    
    # Analyze posture
    posture_counts = student_data['posture_status'].value_counts()
    total = len(student_data)
    bad_posture = posture_counts.get('Bad Posture', 0)
    bad_posture_pct = (bad_posture / total) * 100 if total > 0 else 0
    if bad_posture_pct > 20:
        recommendations.append("Improve sitting posture to maintain better focus and reduce fatigue.")
    
    # Analyze gaze
    gaze_counts = student_data['gaze_status'].value_counts()
    looking_away = gaze_counts.get('Looking left', 0) + gaze_counts.get('Looking right', 0)
    away_pct = (looking_away / total) * 100 if total > 0 else 0
    if away_pct > 30:
        recommendations.append("Try to maintain visual focus on the instructional content.")
    
    # Check for sleepiness/tiredness
    behaviors = student_data['behavior'].value_counts()
    sleeping = behaviors.get('Sleeping', 0)
    if sleeping > 0 or 'TIRED' in student_data['attention_state'].values:
        recommendations.append("Ensure getting adequate rest before class to avoid drowsiness.")
    
    # If no issues found, provide positive reinforcement
    if not recommendations:
        recommendations.append("Continue with current level of engagement and attention.")
    
    return recommendations

def generate_student_report(student_name, student_data, output_dir):
    """Generate a comprehensive report for a single student"""
    # Create student directory
    student_dir = os.path.join(output_dir, f"{student_name.replace(' ', '_')}")
    os.makedirs(student_dir, exist_ok=True)
    
    # Generate charts
    chart_path = os.path.join(student_dir, "attention_chart.png")
    generate_attention_chart(student_data, chart_path)
    
    # Calculate session duration
    if len(student_data) > 0:
        start_time = student_data['timestamp'].min()
        end_time = student_data['timestamp'].max()
        duration = end_time - start_time
        duration_minutes = duration.total_seconds() / 60
    else:
        duration_minutes = 0
    
    # Analyze data
    attention_percentages = calculate_attention_percentages(student_data)
    distraction_analysis = analyze_distractions(student_data)
    posture_analysis = analyze_posture(student_data)
    emotion_analysis = analyze_emotion(student_data)
    gaze_analysis = analyze_gaze(student_data)
    behavior_analysis = analyze_behaviors(student_data)
    recommendations = generate_recommendations(student_data)
    
    # Create the report
    report_path = os.path.join(student_dir, "report.md")
    
    with open(report_path, 'w') as f:
        f.write(f"# Behavior Analysis Report for {student_name}\n\n")
        f.write(f"**Report Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Session Duration:** {duration_minutes:.1f} minutes\n")
        f.write(f"**Total Observations:** {len(student_data)}\n\n")
        
        f.write("## Attention Overview\n\n")
        f.write("![Attention Distribution](attention_chart.png)\n\n")
        
        f.write("### Key Findings\n\n")
        f.write(f"- **Focused Time:** {attention_percentages['High']:.1f}%\n")
        f.write(f"- **Partially Engaged:** {attention_percentages['Medium']:.1f}%\n")
        f.write(f"- **Disengaged:** {attention_percentages['Low']:.1f}%\n\n")
        
        f.write("## Detailed Analysis\n\n")
        
        f.write("### Distraction Analysis\n\n")
        f.write(f"{distraction_analysis}\n\n")
        
        f.write("### Posture Assessment\n\n")
        f.write(f"{posture_analysis}\n\n")
        
        f.write("### Emotional State\n\n")
        f.write(f"{emotion_analysis}\n\n")
        
        f.write("### Gaze Tracking\n\n")
        f.write(f"{gaze_analysis}\n\n")
        
        f.write("### Specific Behaviors\n\n")
        f.write(f"{behavior_analysis}\n\n")
        
        f.write("## Recommendations\n\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
        
        f.write("\n\n")
        f.write("---\n")
        f.write("*This report was automatically generated based on behavioral monitoring data.*\n")
        f.write("*For questions or concerns, please contact the instructor.*\n")
    
    # Also generate a simple summary text file
    summary_path = os.path.join(student_dir, "summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write(f"BEHAVIOR SUMMARY FOR {student_name.upper()}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Session Duration: {duration_minutes:.1f} minutes\n")
        f.write(f"Observations: {len(student_data)}\n\n")
        f.write(f"Attention Levels:\n")
        f.write(f"- Focused: {attention_percentages['High']:.1f}%\n")
        f.write(f"- Partially Engaged: {attention_percentages['Medium']:.1f}%\n")
        f.write(f"- Disengaged: {attention_percentages['Low']:.1f}%\n\n")
        f.write("KEY RECOMMENDATIONS:\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
    
    return report_path

def main():
    """Main function to generate reports for all students"""
    parser = argparse.ArgumentParser(description='Generate student behavior reports from log data')
    parser.add_argument('log_file', help='Path to the behavior log CSV file')
    parser.add_argument('--output', '-o', default='student_reports', help='Output directory for reports')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load and preprocess data
    df = load_log_data(args.log_file)
    if df is None:
        print("Error: Could not load log data. Exiting.")
        return
    
    df = preprocess_data(df)
    
    # Get unique student names
    student_names = get_student_names(df)
    print(f"Found {len(student_names)} students in the log data")
    
    # Generate reports for each student
    for name in student_names:
        if name == "Unknown":
            continue  # Skip unknown faces
            
        print(f"Generating report for {name}...")
        student_data = df[df['name'] == name]
        report_path = generate_student_report(name, student_data, args.output)
        print(f"Report saved to {report_path}")
    
    print(f"All reports generated successfully in {args.output}")

if __name__ == "__main__":
    main()