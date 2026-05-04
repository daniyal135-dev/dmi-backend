from rest_framework import generics, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import ForumThread, ForumPost
from .serializers import ForumThreadSerializer, ForumPostSerializer

class ForumThreadListCreateView(generics.ListCreateAPIView):
    queryset = ForumThread.objects.all()
    serializer_class = ForumThreadSerializer

class ForumPostListCreateView(generics.ListCreateAPIView):
    serializer_class = ForumPostSerializer
    
    def get_queryset(self):
        thread_id = self.kwargs['thread_id']
        return ForumPost.objects.filter(thread_id=thread_id)