from rest_framework import generics, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import PDAEntry
from .serializers import PDAEntrySerializer

class PDAEntryListCreateView(generics.ListCreateAPIView):
    queryset = PDAEntry.objects.filter(approved=True)
    serializer_class = PDAEntrySerializer

@api_view(['POST'])
def submit_to_archive(request):
    # TODO: Implement archive submission logic
    return Response({'message': 'Archive submission endpoint - to be implemented'}, status=status.HTTP_200_OK)