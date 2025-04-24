# gui/callbacks.py

def callback(widget_name):
    """Декоратор: помечает методы контроллера для конкретных виджетов"""
    def decorator(func):
        func._widget_name = widget_name
        return func
    return decorator