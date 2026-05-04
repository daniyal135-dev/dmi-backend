from django.urls import path
from .views import AnalysisResultListCreateView, AnalysisResultDetailView, analyze_image, analyze_video, analyze_text

urlpatterns = [
    path('results/', AnalysisResultListCreateView.as_view(), name='analysis-results'),
    path('results/<int:pk>/', AnalysisResultDetailView.as_view(), name='analysis-result-detail'),
    path('image/', analyze_image, name='analyze-image'),
    path('video/', analyze_video, name='analyze-video'),
    path('text/', analyze_text, name='analyze-text'),
]





