"""
Test script to demonstrate the optimized confidence scoring system
This script shows how the new confidence algorithm works with sample data
"""

def test_distance_to_confidence():
    """Test the new distance-to-confidence conversion function"""
    
    def distance_to_confidence(distance):
        """Convert ChromaDB distance to confidence percentage"""
        if distance < 0.3:      # Excellent match
            return 0.95 + (0.3 - distance) * 0.15  # 95-98%
        elif distance < 0.6:    # Very good match  
            return 0.85 + (0.6 - distance) * 0.33  # 85-95%
        elif distance < 1.0:    # Good match
            return 0.75 + (1.0 - distance) * 0.25  # 75-85%
        elif distance < 1.5:    # Fair match
            return 0.60 + (1.5 - distance) * 0.30  # 60-75%
        else:                   # Poor match
            return max(0.40, 0.70 - (distance - 1.5) * 0.20)  # 40-70%
    
    test_distances = [0.1, 0.2, 0.4, 0.8, 1.2, 1.8, 2.5]
    
    print("🧪 Testing Optimized Confidence Scoring Algorithm")
    print("=" * 60)
    print("Distance Score → Confidence Score")
    print("-" * 35)
    
    for distance in test_distances:
        confidence = distance_to_confidence(distance)
        confidence_percent = confidence * 100
        
        if confidence >= 0.85:
            status = "🟢 Excellent"
        elif confidence >= 0.75:
            status = "🟡 Very Good"
        elif confidence >= 0.60:
            status = "🟠 Good"
        else:
            status = "🔴 Fair/Poor"
            
        print(f"{distance:4.1f}          → {confidence_percent:5.1f}% {status}")
    
    print("\n📊 Quality Boost Simulation")
    print("-" * 30)
    
    base_confidence = 0.78
    quality_indicators = {
        'context_references': True,
        'specific_details': True,
        'structured_response': False,
        'factual_language': True,
        'comprehensive_answer': True
    }
    
    quality_boost = 1.0
    if quality_indicators['context_references']:
        quality_boost *= 1.08
    if quality_indicators['specific_details']:
        quality_boost *= 1.06
    if quality_indicators['structured_response']:
        quality_boost *= 1.04
    if quality_indicators['factual_language']:
        quality_boost *= 1.05
    if quality_indicators['comprehensive_answer']:
        quality_boost *= 1.07
    
    enhanced_confidence = base_confidence * quality_boost
    
    print(f"Base Confidence: {base_confidence*100:.1f}%")
    print(f"Quality Boost: {quality_boost:.3f}x")
    print(f"Enhanced Confidence: {enhanced_confidence*100:.1f}%")
    
    print("\n✨ Key Improvements:")
    print("• Distance-based confidence scoring (95-98% for excellent matches)")
    print("• Multi-document relevance boosting")
    print("• Quality indicator enhancements")  
    print("• Context coverage analysis")
    print("• Intelligent confidence thresholds")
    print("\n🎯 Expected Result: >80% confidence for relevant queries!")

if __name__ == "__main__":
    test_distance_to_confidence()
