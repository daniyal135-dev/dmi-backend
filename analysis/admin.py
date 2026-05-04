from django.contrib import admin
from analysis.models import AnalysisResult
from forum.models import ForumThread, ForumPost
from archive.models import PDAEntry

@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ['user', 'file_type', 'verdict', 'confidence', 'created_at']
    list_filter = ['file_type', 'verdict', 'created_at']
    search_fields = ['user__username']

@admin.register(ForumThread)
class ForumThreadAdmin(admin.ModelAdmin):
    list_display = ['title', 'author', 'votes', 'created_at']
    list_filter = ['created_at']
    search_fields = ['title', 'content']

@admin.register(ForumPost)
class ForumPostAdmin(admin.ModelAdmin):
    list_display = ['thread', 'author', 'votes', 'created_at']
    list_filter = ['created_at']
    search_fields = ['content']

@admin.register(PDAEntry)
class PDAEntryAdmin(admin.ModelAdmin):
    list_display = ['title', 'category', 'submitted_by', 'approved', 'created_at']
    list_filter = ['category', 'approved', 'created_at']
    search_fields = ['title', 'description']