from flask import Blueprint, request, jsonify, session
from session_manager import SessionManager
from models import db, Memory, UserSession

memory_api = Blueprint('memory_api', __name__)

@memory_api.route('/api/memories', methods=['GET'])
def get_memories():
    """Get memories for current session"""
    current_session = SessionManager.get_current_session()
    if not current_session:
        return jsonify({'error': 'No active session'}), 401
    
    memory_type = request.args.get('type')
    limit = int(request.args.get('limit', 50))
    
    memories = SessionManager.get_memories(
        session_id=current_session.id,
        memory_type=memory_type,
        limit=limit
    )
    
    return jsonify({
        'memories': [{
            'id': m.id,
            'type': m.memory_type,
            'content': m.content,
            'metadata': m.memory_metadata,
            'created_at': m.created_at.isoformat(),
            'importance_score': m.importance_score
        } for m in memories]
    })

@memory_api.route('/api/memories', methods=['POST'])
def store_memory():
    """Store a new memory for current session"""
    current_session = SessionManager.get_current_session()
    if not current_session:
        return jsonify({'error': 'No active session'}), 401
    
    data = request.get_json()
    memory_type = data.get('type', 'general')
    content = data.get('content')
    metadata = data.get('metadata', {})
    importance_score = data.get('importance_score', 0.5)
    
    if not content:
        return jsonify({'error': 'Content is required'}), 400
    
    memory = SessionManager.store_memory(
        session_id=current_session.id,
        memory_type=memory_type,
        content=content,
        metadata=metadata,
        importance_score=importance_score
    )
    
    return jsonify({
        'success': True,
        'memory': {
            'id': memory.id,
            'type': memory.memory_type,
            'content': memory.content,
            'created_at': memory.created_at.isoformat()
        }
    })

@memory_api.route('/api/memories/accessible', methods=['GET'])
def get_accessible_memories():
    """Get all memories accessible to current session based on access level"""
    current_session = SessionManager.get_current_session()
    if not current_session:
        return jsonify({'error': 'No active session'}), 401
    
    memories = SessionManager.get_accessible_memories(current_session)
    
    return jsonify({
        'memories': [{
            'id': m.id,
            'type': m.memory_type,
            'content': m.content,
            'metadata': m.memory_metadata,
            'created_at': m.created_at.isoformat(),
            'importance_score': m.importance_score,
            'session_id': m.session_id
        } for m in memories]
    })

@memory_api.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all sessions for current user"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'No active session'}), 401
    
    user_sessions = UserSession.query.filter_by(
        user_id=user_id,
        is_active=True
    ).order_by(UserSession.last_activity.desc()).all()
    
    return jsonify({
        'sessions': [{
            'id': s.id,
            'login_time': s.login_time.isoformat(),
            'last_activity': s.last_activity.isoformat(),
            'ip_address': s.ip_address,
            'access_level': s.access_level,
            'session_notes': s.session_notes,
            'memory_count': len(s.memories)
        } for s in user_sessions]
    })

@memory_api.route('/api/sessions/<int:session_id>/memories', methods=['GET'])
def get_session_memories(session_id):
    """Get memories for a specific session"""
    current_session = SessionManager.get_current_session()
    if not current_session:
        return jsonify({'error': 'No active session'}), 401
    
    # Check if user has access to this session
    target_session = UserSession.query.filter_by(
        id=session_id,
        user_id=current_session.user_id
    ).first()
    
    if not target_session:
        return jsonify({'error': 'Session not found or access denied'}), 404
    
    memories = SessionManager.get_memories(session_id=session_id)
    
    return jsonify({
        'session': {
            'id': target_session.id,
            'login_time': target_session.login_time.isoformat(),
            'access_level': target_session.access_level,
            'session_notes': target_session.session_notes
        },
        'memories': [{
            'id': m.id,
            'type': m.memory_type,
            'content': m.content,
            'metadata': m.memory_metadata,
            'created_at': m.created_at.isoformat(),
            'importance_score': m.importance_score
        } for m in memories]
    })

@memory_api.route('/api/memories/search', methods=['GET'])
def search_memories():
    """Search memories by content"""
    current_session = SessionManager.get_current_session()
    if not current_session:
        return jsonify({'error': 'No active session'}), 401
    
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Search query required'}), 400
    
    # Search in accessible memories
    memories = SessionManager.get_accessible_memories(current_session)
    
    # Filter by search query
    filtered_memories = [
        m for m in memories 
        if query.lower() in m.content.lower() or 
           query.lower() in m.memory_type.lower()
    ]
    
    return jsonify({
        'query': query,
        'memories': [{
            'id': m.id,
            'type': m.memory_type,
            'content': m.content,
            'metadata': m.memory_metadata,
            'created_at': m.created_at.isoformat(),
            'importance_score': m.importance_score
        } for m in filtered_memories]
    })

@memory_api.route('/api/memories/<int:memory_id>', methods=['DELETE'])
def delete_memory(memory_id):
    """Delete a memory (only if user owns it)"""
    current_session = SessionManager.get_current_session()
    if not current_session:
        return jsonify({'error': 'No active session'}), 401
    
    memory = Memory.query.filter_by(
        id=memory_id,
        session_id=current_session.id
    ).first()
    
    if not memory:
        return jsonify({'error': 'Memory not found or access denied'}), 404
    
    db.session.delete(memory)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Memory deleted'})
