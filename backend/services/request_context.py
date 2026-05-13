from contextvars import ContextVar

_REQUEST_ID = ContextVar("request_id", default=None)


def set_request_id(request_id):
    return _REQUEST_ID.set(request_id)


def get_request_id():
    return _REQUEST_ID.get()


def reset_request_id(token):
    _REQUEST_ID.reset(token)
