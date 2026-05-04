from django.urls import path
from .views import ForumThreadListCreateView, ForumPostListCreateView

urlpatterns = [
    path('threads/', ForumThreadListCreateView.as_view(), name='forum-threads'),
    path('threads/<int:thread_id>/posts/', ForumPostListCreateView.as_view(), name='forum-posts'),
]





