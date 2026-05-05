from rest_framework import serializers
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'role', 'reputation', 'created_at', 'last_login']
        read_only_fields = ['id', 'created_at', 'last_login']

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    """
    Custom token serializer that allows login with username OR email.
    """
    def validate(self, attrs):
        # Get the username field (could be username or email)
        username_or_email = attrs.get('username', '')
        
        # Check if it's an email (contains @)
        if '@' in username_or_email:
            qs = User.objects.filter(email__iexact=username_or_email)
            n = qs.count()
            if n == 1:
                attrs['username'] = qs.first().username
            elif n > 1:
                raise serializers.ValidationError(
                    {'username': 'Multiple accounts use this email; sign in with your username.'}
                )
        
        # Call parent validation
        return super().validate(attrs)
    
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token['username'] = user.username
        token['role'] = user.role
        return token





