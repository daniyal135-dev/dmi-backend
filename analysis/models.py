from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class AnalysisResult(models.Model):
    FILE_TYPE_CHOICES = [
        ('image', 'Image'),
        ('video', 'Video'),
        ('text', 'Text'),
    ]
    
    VERDICT_CHOICES = [
        ('real', 'Real'),
        ('fake', 'Fake'),
        ('uncertain', 'Uncertain'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='analyses')
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES)
    file_path = models.CharField(max_length=500)
    verdict = models.CharField(max_length=10, choices=VERDICT_CHOICES)
    confidence = models.FloatField()
    heatmap_path = models.CharField(max_length=500, blank=True, null=True)
    metadata = models.JSONField(default=dict)
    explanation = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.file_type} - {self.verdict}"