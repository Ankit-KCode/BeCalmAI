
from django.contrib import admin
from django.urls import path
from BeCalmAI.views import chat_view, chat_api

urlpatterns = [
    path('', chat_view, name='chat'),
    path('admin/', admin.site.urls),
    path('BeCalmAI/', chat_view, name='chat'),
    path('chat-api/', chat_api, name='chat_api'),  # API for chat responses
]
