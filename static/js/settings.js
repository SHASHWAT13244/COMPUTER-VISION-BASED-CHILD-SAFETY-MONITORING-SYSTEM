{
    "enabled_methods": ["desktop", "email"],
    "email": {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": "your_email@gmail.com",
        "sender_password": "your_app_password",
        "recipient_emails": ["recipient1@email.com", "recipient2@email.com"]
    },
    "sms": {
        "api_key": "your_twilio_api_key",
        "account_sid": "your_account_sid",
        "auth_token": "your_auth_token",
        "from_number": "+1234567890",
        "phone_numbers": ["+1234567890"]
    },
    "telegram": {
        "bot_token": "your_bot_token",
        "chat_ids": ["chat_id_1", "chat_id_2"]
    },
    "webhook": {
        "urls": ["https://your-webhook.com/endpoint"]
    },
    "desktop": {
        "enable_sound": true,
        "enable_popup": true
    },
    "throttling": {
        "max_alerts_per_minute": 10,
        "cooldown_seconds": 5
    }
}