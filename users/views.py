from rest_framework import generics, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.views import TokenObtainPairView
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError as DjangoValidationError
from django.db import IntegrityError, connection
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from .serializers import UserSerializer, CustomTokenObtainPairSerializer

User = get_user_model()


@require_GET
def health_check(request):
    """JSON health — DB reachable + users table (no auth). Use for Railway / Postman."""
    checks = {}
    errors = []
    try:
        connection.ensure_connection()
        checks["database_connection"] = "ok"
    except Exception as e:
        checks["database_connection"] = "failed"
        errors.append({"step": "ensure_connection", "error": repr(e)})

    if checks.get("database_connection") == "ok":
        try:
            User.objects.exists()
            checks["users_query"] = "ok"
        except Exception as e:
            checks["users_query"] = "failed"
            errors.append({"step": "users_query", "error": repr(e)})

    ok = (
        checks.get("database_connection") == "ok"
        and checks.get("users_query") == "ok"
    )
    payload = {"status": "ok" if ok else "error", "checks": checks}
    if errors:
        payload["errors"] = errors
    return JsonResponse(payload, status=200 if ok else 503)


class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer

@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    serializer = UserSerializer(data=request.data)
    if serializer.is_valid():
        try:
            user = User.objects.create_user(
                username=serializer.validated_data['username'],
                email=serializer.validated_data['email'],
                password=request.data['password'],
            )
        except DjangoValidationError as e:
            return Response(
                {'error': e.messages if hasattr(e, 'messages') else str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except IntegrityError:
            return Response(
                {'error': 'Username or email is already registered.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        return Response(UserSerializer(user).data, status=status.HTTP_201_CREATED)
    errors = serializer.errors
    # Clear message when username is already taken
    if 'username' in errors:
        msg = errors['username']
        if isinstance(msg, list) and msg and 'already exists' in str(msg[0]).lower():
            return Response(
                {'error': 'This username is already taken. Please choose a different username.'},
                status=status.HTTP_400_BAD_REQUEST
            )
    return Response(errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def profile(request):
    serializer = UserSerializer(request.user)
    return Response(serializer.data)