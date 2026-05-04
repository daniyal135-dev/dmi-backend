from rest_framework import serializers
from .models import PDAEntry

class PDAEntrySerializer(serializers.ModelSerializer):
    submitted_by = serializers.StringRelatedField(read_only=True)
    
    class Meta:
        model = PDAEntry
        fields = ['id', 'title', 'description', 'category', 'file_path', 'analysis_data', 'submitted_by', 'approved', 'created_at']
        read_only_fields = ['id', 'submitted_by', 'created_at']





