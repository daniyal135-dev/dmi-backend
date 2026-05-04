from django.urls import path
from .views import PDAEntryListCreateView, submit_to_archive

urlpatterns = [
    path('entries/', PDAEntryListCreateView.as_view(), name='archive-entries'),
    path('submit/', submit_to_archive, name='submit-to-archive'),
]





