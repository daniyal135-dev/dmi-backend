from rest_framework import serializers
from .models import AnalysisResult
from django.contrib.auth import get_user_model

User = get_user_model()

class AnalysisResultSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    username = serializers.CharField(source='user.username', read_only=True)
    
    class Meta:
        model = AnalysisResult
        fields = ['id', 'user', 'username', 'file_type', 'file_path', 'verdict', 
                  'confidence', 'heatmap_path', 'metadata', 'explanation', 
                  'created_at', 'updated_at']
        read_only_fields = ['id', 'user', 'username', 'created_at', 'updated_at']

class AnalysisCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating analysis results"""
    class Meta:
        model = AnalysisResult
        fields = ['file_type', 'file_path', 'verdict', 'confidence', 
                  'heatmap_path', 'metadata', 'explanation']





