from flask import jsonify, has_app_context

def _safe_response(data: dict, status_code: int):
    """
    Return a Flask JSON response if inside app context,
    otherwise return a plain Python dict (for testing/utility usage).
    """
    if has_app_context():
        return jsonify(data), status_code
    return data  # no jsonify if outside Flask

## Default message response
def message(status_code: int, message: str):
    data = {
        "status_code": status_code,
        "message": message,
    }
    return _safe_response(data, status_code)

## Error Message response
def message_error(status_code: int, error: str, message: str):
    data = {
        "status_code": status_code,
        "message": message,
        "error": error,
    }
    return _safe_response(data, status_code)

## Custom Message response
def message_custom(data: dict, status_code: int, message: str):
    data['status_code'] = status_code
    data['message'] = message
    return _safe_response(data, status_code)
