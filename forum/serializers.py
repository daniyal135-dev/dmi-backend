from rest_framework import serializers
from .models import ForumThread, ForumPost

class ForumThreadSerializer(serializers.ModelSerializer):
    author = serializers.StringRelatedField(read_only=True)
    
    class Meta:
        model = ForumThread
        fields = ['id', 'title', 'content', 'author', 'votes', 'created_at']
        read_only_fields = ['id', 'author', 'votes', 'created_at']

class ForumPostSerializer(serializers.ModelSerializer):
    author = serializers.StringRelatedField(read_only=True)
    
    class Meta:
        model = ForumPost
        fields = ['id', 'content', 'author', 'votes', 'created_at']
        read_only_fields = ['id', 'author', 'votes', 'created_at']





