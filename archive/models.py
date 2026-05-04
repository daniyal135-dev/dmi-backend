from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class PDAEntry(models.Model):
    CATEGORY_CHOICES = [
        ('political', 'Political'),
        ('celebrity', 'Celebrity'),
        ('news', 'News'),
        ('entertainment', 'Entertainment'),
        ('other', 'Other'),
    ]
    
    title = models.CharField(max_length=200)
    description = models.TextField()
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    file_path = models.CharField(max_length=500)
    analysis_data = models.JSONField(default=dict)
    submitted_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='submissions')
    approved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title