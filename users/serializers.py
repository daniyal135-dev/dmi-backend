from rest_framework import serializers
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'role', 'reputation', 'created_at']
        read_only_fields = ['id', 'created_at']

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    """
    Custom token serializer that allows login with username OR email.
    """
    def validate(self, attrs):
        # Get the username field (could be username or email)
        username_or_email = attrs.get('username', '')
        
        # Check if it's an email (contains @)
        if '@' in username_or_email:
            # Try to find user by email
            try:
                user = User.objects.get(email=username_or_email)
                # Replace email with actual username for authentication
                attrs['username'] = user.username
            except User.DoesNotExist:
                pass  # Let it fail naturally with "invalid credentials"
        
        # Call parent validation
        return super().validate(attrs)
    
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token['username'] = user.username
        token['role'] = user.role
        return token





