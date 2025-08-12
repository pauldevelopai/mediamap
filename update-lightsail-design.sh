#!/bin/bash

# Your existing Lightsail instance details
INSTANCE_IP="13.40.124.51"
SSH_KEY="LightsailDefaultKey-eu-west-2.pem"

echo "🎨 Updating DataSafe design on Lightsail instance..."
echo "🌐 IP Address: $INSTANCE_IP"

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found: $SSH_KEY"
    echo "💡 Please download your Lightsail SSH key first"
    exit 1
fi

echo "🔑 Using SSH key: $SSH_KEY"

# Create a temporary file with the updated template
echo "📝 Creating updated template..."
cat > /tmp/user_dashboard_updated.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataSafe AI - Your Business Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.2/font/bootstrap-icons.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #6366f1;
            --primary-dark: #4f46e5;
            --secondary-color: #f59e0b;
            --success-color: #10b981;
            --danger-color: #ef4444;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-600: #4b5563;
            --gray-700: #374151;
            --gray-800: #1f2937;
            --gray-900: #111827;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: var(--gray-800);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            text-rendering: optimizeLegibility;
        }

        /* Header */
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--gray-200);
            padding: 1rem 0;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 2rem;
        }

        .brand {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary-color);
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .user-actions {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .btn-secondary {
            background: var(--gray-100);
            color: var(--gray-700);
            border: 1px solid var(--gray-200);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            text-decoration: none;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-secondary:hover {
            background: var(--gray-200);
            color: var(--gray-800);
            text-decoration: none;
        }

        .btn-primary {
            background: var(--primary-color);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-primary:hover {
            background: var(--primary-dark);
            transform: translateY(-1px);
        }

        /* Main Container */
        .main-container {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 2rem;
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 2rem;
            min-height: calc(100vh - 120px);
        }

        /* Chat Section */
        .chat-section {
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .chat-header {
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            color: white;
            padding: 2rem;
            text-align: center;
        }

        .chat-header h1 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .chat-header p {
            opacity: 0.9;
            font-size: 0.875rem;
        }

        .chat-body {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 1.5rem;
        }

        .chat-messages {
            flex: 1;
            border: 1px solid var(--gray-200);
            border-radius: 12px;
            padding: 1.5rem;
            background: var(--gray-50);
            overflow-y: auto;
            margin-bottom: 1.5rem;
            max-height: 400px;
            -webkit-overflow-scrolling: touch;
            scroll-behavior: smooth;
        }

        .message {
            margin-bottom: 1rem;
            padding: 1rem;
            border-radius: 12px;
            max-width: 85%;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }

        .user-message {
            background: var(--primary-color);
            color: white;
            margin-left: auto;
            text-align: right;
        }

        .ai-message {
            background: white;
            border: 1px solid var(--gray-200);
            color: var(--gray-800);
        }

        .chat-input-section {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .chat-input {
            display: flex;
            gap: 1rem;
            align-items: flex-end;
        }

        .chat-input input {
            flex: 1;
            border: 2px solid var(--gray-200);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            font-size: 1rem;
            outline: none;
            transition: all 0.2s;
            background: white;
        }

        .chat-input input:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }

        .chat-input button {
            background: var(--primary-color);
            border: none;
            border-radius: 12px;
            color: white;
            padding: 1rem 2rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .chat-input button:hover {
            background: var(--primary-dark);
            transform: translateY(-1px);
        }

        .action-buttons {
            display: flex;
            gap: 1rem;
        }

        .action-btn {
            flex: 1;
            background: var(--gray-100);
            border: 1px solid var(--gray-200);
            border-radius: 10px;
            color: var(--gray-700);
            padding: 0.75rem 1rem;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .action-btn:hover {
            background: var(--gray-200);
            transform: translateY(-1px);
        }

        .action-btn.primary {
            background: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }

        .action-btn.primary:hover {
            background: var(--primary-dark);
        }

        .action-btn.secondary {
            background: var(--secondary-color);
            color: white;
            border-color: var(--secondary-color);
        }

        .action-btn.secondary:hover {
            background: #d97706;
        }

        /* Sidebar */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .info-card {
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }

        .info-card-header {
            background: var(--gray-50);
            padding: 1.5rem;
            border-bottom: 1px solid var(--gray-200);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .info-card-title {
            font-size: 1rem;
            font-weight: 600;
            color: var(--gray-800);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .info-card-actions {
            display: flex;
            gap: 0.5rem;
        }

        .btn-icon {
            background: var(--gray-100);
            border: 1px solid var(--gray-200);
            color: var(--gray-600);
            border-radius: 6px;
            padding: 0.375rem;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .btn-icon:hover {
            background: var(--gray-200);
            color: var(--gray-800);
        }

        .btn-icon.download {
            background: #dbeafe;
            color: #1d4ed8;
            border-color: #bfdbfe;
        }

        .btn-icon.download:hover {
            background: #bfdbfe;
        }

        .btn-icon.save {
            background: #dcfce7;
            color: #15803d;
            border-color: #bbf7d0;
        }

        .btn-icon.save:hover {
            background: #bbf7d0;
        }

        .btn-icon.clear {
            background: #fee2e2;
            color: #dc2626;
            border-color: #fecaca;
        }

        .btn-icon.clear:hover {
            background: #fecaca;
        }

        .info-card-body {
            padding: 1.5rem;
            min-height: 200px;
        }

        .info-placeholder {
            color: var(--gray-600);
            font-size: 0.875rem;
            text-align: center;
            padding: 2rem 1rem;
        }

        .info-content {
            background: var(--gray-50);
            border: 1px solid var(--gray-200);
            border-radius: 8px;
            padding: 1rem;
            max-height: 300px;
            overflow-y: auto;
            font-size: 0.875rem;
            line-height: 1.5;
            -webkit-overflow-scrolling: touch;
            scroll-behavior: smooth;
        }

        .info-content::-webkit-scrollbar {
            width: 6px;
        }

        .info-content::-webkit-scrollbar-track {
            background: var(--gray-100);
            border-radius: 3px;
        }

        .info-content::-webkit-scrollbar-thumb {
            background: var(--gray-300);
            border-radius: 3px;
        }

        .info-content::-webkit-scrollbar-thumb:hover {
            background: var(--gray-400);
        }

        /* Responsive Design */
        @media (max-width: 1024px) {
            .main-container {
                grid-template-columns: 1fr;
                gap: 1.5rem;
            }

            .sidebar {
                order: -1;
            }
        }

        @media (max-width: 768px) {
            .header-content {
                padding: 0 1rem;
                flex-direction: column;
                gap: 1rem;
            }

            .user-actions {
                flex-wrap: wrap;
                justify-content: center;
            }

            .main-container {
                margin: 1rem auto;
                padding: 0 1rem;
            }

            .chat-header {
                padding: 1.5rem;
            }

            .chat-header h1 {
                font-size: 1.25rem;
            }

            .chat-body {
                padding: 1rem;
            }

            .chat-messages {
                max-height: 300px;
            }

            .chat-input {
                flex-direction: column;
                align-items: stretch;
            }

            .action-buttons {
                flex-direction: column;
            }

            .info-card-header {
                padding: 1rem;
                flex-direction: column;
                gap: 1rem;
                align-items: stretch;
            }

            .info-card-actions {
                justify-content: center;
            }
        }

        @media (max-width: 480px) {
            .header-content {
                padding: 0 0.5rem;
            }

            .main-container {
                padding: 0 0.5rem;
                margin: 0.5rem auto;
            }

            .chat-header {
                padding: 1rem;
            }

            .chat-body {
                padding: 0.75rem;
            }

            .info-card-body {
                padding: 1rem;
            }

            .btn-secondary, .btn-primary {
                padding: 0.375rem 0.75rem;
                font-size: 0.8rem;
            }
        }

        /* Loading states */
        .loading {
            opacity: 0.6;
            pointer-events: none;
        }

        /* Success animations */
        .success-animation {
            animation: successPulse 0.5s ease-in-out;
        }

        @keyframes successPulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        /* Performance optimizations */
        .chat-messages, .info-content {
            will-change: scroll-position;
        }

        .btn-icon, .action-btn, .chat-input button {
            will-change: transform;
        }

        @media (prefers-reduced-motion: no-preference) {
            .btn-icon:hover, .action-btn:hover, .chat-input button:hover {
                transform: translateY(-1px);
            }
        }

        @supports not (backdrop-filter: blur(10px)) {
            .header {
                background: rgba(255, 255, 255, 0.98);
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="header-content">
            <a href="#" class="brand">
                <i class="bi bi-robot"></i>
                DataSafe AI
            </a>
            <div class="user-actions">
                <button class="btn-secondary" data-bs-toggle="modal" data-bs-target="#feedbackModal">
                    <i class="bi bi-chat-quote"></i>
                    Feedback
                </button>
                <a href="{{ url_for('my_chats') }}" class="btn-secondary">
                    <i class="bi bi-chat-text"></i>
                    My Chats
                </a>
                <a href="{{ url_for('logout') }}" class="btn-primary">
                    <i class="bi bi-box-arrow-right"></i>
                    Logout
                </a>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <div class="main-container">
        <!-- Chat Section -->
        <section class="chat-section">
            <div class="chat-header">
                <h1><i class="bi bi-chat-dots me-2"></i>Highlander AI Business Consultant</h1>
                <p>Tell me about your media business and discover how AI can help you grow!</p>
            </div>
            
            <div class="chat-body">
                <div id="chat-messages" class="chat-messages">
                    <div class="message ai-message">
                        <strong><i class="bi bi-robot me-2"></i>Highlander:</strong><br>
                        I'm Highlander, your AI business consultant. Tell me about your media business - what challenges are you facing and what are you trying to achieve?
                    </div>
                </div>
                
                <div class="chat-input-section">
                    <form id="chat-form" class="chat-input" autocomplete="off">
                        <input type="text" id="chat-input" placeholder="Describe your business, challenges, or goals..." required />
                        <button type="submit">
                            <i class="bi bi-send"></i>
                            Send
                        </button>
                    </form>
                    
                    <div class="action-buttons">
                        <button id="extract-facts-btn" class="action-btn primary">
                            <i class="bi bi-file-text"></i>
                            Extract Company Info
                        </button>
                        <button id="develop-strategies-btn" class="action-btn secondary">
                            <i class="bi bi-lightbulb"></i>
                            Develop AI Strategies
                        </button>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Sidebar -->
        <aside class="sidebar">
            <!-- Company Information Card -->
            <div class="info-card" id="fact-sheet-box">
                <div class="info-card-header">
                    <h3 class="info-card-title">
                        <i class="bi bi-building"></i>
                        Company Information
                    </h3>
                    <div class="info-card-actions">
                        <button class="btn-icon download" id="download-company-btn" onclick="downloadCompanyInfo()" style="display: none;" title="Download">
                            <i class="bi bi-download"></i>
                        </button>
                        <button class="btn-icon save" id="save-company-btn" onclick="saveCompanyInfoToDB()" style="display: none;" title="Save to Database">
                            <i class="bi bi-database"></i>
                        </button>
                        <button class="btn-icon clear" onclick="clearCompanyInfo()" title="Clear">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </div>
                <div class="info-card-body">
                    <p class="info-placeholder" id="company-info-placeholder">
                        Start chatting to extract your company's key information automatically.
                    </p>
                    <div id="company-info-content" class="info-content" style="display: none;"></div>
                </div>
            </div>
            
            <!-- AI Strategies Card -->
            <div class="info-card" id="strategies-box">
                <div class="info-card-header">
                    <h3 class="info-card-title">
                        <i class="bi bi-gear"></i>
                        AI Strategies
                    </h3>
                    <div class="info-card-actions">
                        <button class="btn-icon download" id="download-strategies-btn" onclick="downloadStrategies()" style="display: none;" title="Download">
                            <i class="bi bi-download"></i>
                        </button>
                        <button class="btn-icon save" id="save-strategies-btn" onclick="saveStrategiesToDB()" style="display: none;" title="Save to Database">
                            <i class="bi bi-database"></i>
                        </button>
                        <button class="btn-icon clear" onclick="clearStrategies()" title="Clear">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </div>
                <div class="info-card-body">
                    <p class="info-placeholder" id="strategies-placeholder">
                        AI-powered strategies will appear here based on your business needs.
                    </p>
                    <div id="strategies-content" class="info-content" style="display: none;"></div>
                </div>
            </div>
        </aside>
    </div>

    <script>
        const chatForm = document.getElementById('chat-form');
        const chatInput = document.getElementById('chat-input');
        const chatMessages = document.getElementById('chat-messages');
        const extractFactsBtn = document.getElementById('extract-facts-btn');
        const developStrategiesBtn = document.getElementById('develop-strategies-btn');
        const factSheetBox = document.getElementById('fact-sheet-box');
        const strategiesBox = document.getElementById('strategies-box');
        const companyInfoPlaceholder = document.getElementById('company-info-placeholder');
        const companyInfoContent = document.getElementById('company-info-content');
        const strategiesPlaceholder = document.getElementById('strategies-placeholder');
        const strategiesContent = document.getElementById('strategies-content');
        let chatId = null;

        // Auto-focus on input
        chatInput.focus();

        // Download functions
        function downloadCompanyInfo() {
            const content = companyInfoContent.querySelector('div div');
            if (content && content.textContent.trim()) {
                const blob = new Blob([content.textContent], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'company_information.txt';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                showSuccessMessage('download-company-btn');
            }
        }

        function downloadStrategies() {
            const content = strategiesContent.querySelector('div div');
            if (content && content.textContent.trim()) {
                const blob = new Blob([content.textContent], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'ai_strategies.txt';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                showSuccessMessage('download-strategies-btn');
            }
        }

        function showSuccessMessage(buttonId) {
            const btn = document.getElementById(buttonId);
            const originalHTML = btn.innerHTML;
            btn.innerHTML = '<i class="bi bi-check"></i>';
            btn.classList.add('success-animation');
            setTimeout(() => {
                btn.innerHTML = originalHTML;
                btn.classList.remove('success-animation');
            }, 1000);
        }

        // Save to database functions
        async function saveCompanyInfoToDB() {
            const content = companyInfoContent.querySelector('div div');
            if (content && content.textContent.trim()) {
                const saveBtn = document.getElementById('save-company-btn');
                saveBtn.classList.add('loading');
                
                try {
                    const response = await fetch('/save_company_info', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            content: content.textContent.trim(),
                            type: 'company_info'
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        showSuccessMessage('save-company-btn');
                    } else {
                        throw new Error(data.error || 'Failed to save');
                    }
                } catch (error) {
                    console.error('Error saving to database:', error);
                    const saveBtn = document.getElementById('save-company-btn');
                    saveBtn.innerHTML = '<i class="bi bi-x"></i>';
                    setTimeout(() => {
                        saveBtn.innerHTML = '<i class="bi bi-database"></i>';
                    }, 2000);
                } finally {
                    saveBtn.classList.remove('loading');
                }
            }
        }

        async function saveStrategiesToDB() {
            const content = strategiesContent.querySelector('div div');
            if (content && content.textContent.trim()) {
                const saveBtn = document.getElementById('save-strategies-btn');
                saveBtn.classList.add('loading');
                
                try {
                    const response = await fetch('/save_strategies', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            content: content.textContent.trim(),
                            type: 'strategies'
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        showSuccessMessage('save-strategies-btn');
                    } else {
                        throw new Error(data.error || 'Failed to save');
                    }
                } catch (error) {
                    console.error('Error saving to database:', error);
                    const saveBtn = document.getElementById('save-strategies-btn');
                    saveBtn.innerHTML = '<i class="bi bi-x"></i>';
                    setTimeout(() => {
                        saveBtn.innerHTML = '<i class="bi bi-database"></i>';
                    }, 2000);
                } finally {
                    saveBtn.classList.remove('loading');
                }
            }
        }

        // Clear functions
        function clearCompanyInfo() {
            if (companyInfoContent.style.display !== 'none' && companyInfoContent.innerHTML.trim() !== '') {
                if (confirm('Are you sure you want to clear the company information? This action cannot be undone.')) {
                    companyInfoContent.style.display = 'none';
                    companyInfoContent.innerHTML = '';
                    companyInfoPlaceholder.style.display = 'block';
                    document.getElementById('download-company-btn').style.display = 'none';
                    document.getElementById('save-company-btn').style.display = 'none';
                    localStorage.removeItem('datasafe_fact_sheet');
                }
            }
        }

        function clearStrategies() {
            if (strategiesContent.style.display !== 'none' && strategiesContent.innerHTML.trim() !== '') {
                if (confirm('Are you sure you want to clear the AI strategies? This action cannot be undone.')) {
                    strategiesContent.style.display = 'none';
                    strategiesContent.innerHTML = '';
                    strategiesPlaceholder.style.display = 'block';
                    document.getElementById('download-strategies-btn').style.display = 'none';
                    document.getElementById('save-strategies-btn').style.display = 'none';
                    localStorage.removeItem('datasafe_strategies');
                }
            }
        }

        // Chat form submission
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const userMsg = chatInput.value.trim();
            if (!userMsg) return;
            
            // Add user message
            const userDiv = document.createElement('div');
            userDiv.className = 'message user-message';
            userDiv.innerHTML = `<strong><i class="bi bi-person me-2"></i>You:</strong><br>${userMsg}`;
            chatMessages.appendChild(userDiv);
            
            chatInput.value = '';
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: userMsg, chat_id: chatId })
                });
                
                const data = await res.json();
                
                if (data.success) {
                    chatId = data.chat_id || chatId;
                    
                    // Add AI response
                    const aiDiv = document.createElement('div');
                    aiDiv.className = 'message ai-message';
                    aiDiv.innerHTML = `<strong><i class="bi bi-robot me-2"></i>Highlander:</strong><br>${data.reply}`;
                    chatMessages.appendChild(aiDiv);
                } else {
                    // Add error message
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'message ai-message';
                    errorDiv.innerHTML = `<strong><i class="bi bi-exclamation-triangle me-2"></i>Error:</strong><br>Sorry, there was an error. Please try again.`;
                    chatMessages.appendChild(errorDiv);
                }
            } catch (error) {
                console.error('Chat error:', error);
                const errorDiv = document.createElement('div');
                errorDiv.className = 'message ai-message';
                errorDiv.innerHTML = `<strong><i class="bi bi-exclamation-triangle me-2"></i>Error:</strong><br>Connection error. Please check your internet and try again.`;
                chatMessages.appendChild(errorDiv);
            }
            
            chatMessages.scrollTop = chatMessages.scrollHeight;
        });

        // Extract facts button
        extractFactsBtn.addEventListener('click', async () => {
            const originalText = extractFactsBtn.innerHTML;
            extractFactsBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Extracting...';
            extractFactsBtn.disabled = true;
            
            try {
                const response = await fetch('/extract_facts', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ chat_id: chatId })
                });
                
                const data = await response.json();
                
                if (data.success && data.fact_sheet) {
                    companyInfoPlaceholder.style.display = 'none';
                    companyInfoContent.style.display = 'block';
                    companyInfoContent.innerHTML = `
                        <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6;">
                            <div style="white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; margin: 0; font-size: 0.875rem; line-height: 1.5;">${data.fact_sheet}</div>
                        </div>
                    `;
                    document.getElementById('download-company-btn').style.display = 'inline-block';
                    document.getElementById('save-company-btn').style.display = 'inline-block';
                    localStorage.setItem('datasafe_fact_sheet', data.fact_sheet);
                } else {
                    alert('No conversation found to extract facts from. Please chat with Highlander first.');
                }
            } catch (error) {
                console.error('Error extracting facts:', error);
                alert('Error extracting facts. Please try again.');
            } finally {
                extractFactsBtn.innerHTML = originalText;
                extractFactsBtn.disabled = false;
            }
        });

        // Develop strategies button
        developStrategiesBtn.addEventListener('click', async () => {
            const originalText = developStrategiesBtn.innerHTML;
            developStrategiesBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Developing...';
            developStrategiesBtn.disabled = true;
            
            try {
                const response = await fetch('/develop_strategies', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ chat_id: chatId })
                });
                
                const data = await response.json();
                
                if (data.success && data.strategies) {
                    strategiesPlaceholder.style.display = 'none';
                    strategiesContent.style.display = 'block';
                    strategiesContent.innerHTML = `
                        <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; border-left: 4px solid #f59e0b;">
                            <div style="white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; margin: 0; font-size: 0.875rem; line-height: 1.5;">${data.strategies}</div>
                        </div>
                    `;
                    document.getElementById('download-strategies-btn').style.display = 'inline-block';
                    document.getElementById('save-strategies-btn').style.display = 'inline-block';
                    localStorage.setItem('datasafe_strategies', data.strategies);
                } else {
                    alert('No conversation found to develop strategies from. Please chat with Highlander first.');
                }
            } catch (error) {
                console.error('Error developing strategies:', error);
                alert('Error developing strategies. Please try again.');
            } finally {
                developStrategiesBtn.innerHTML = originalText;
                developStrategiesBtn.disabled = false;
            }
        });

        // Load saved data on page load
        window.addEventListener('load', async () => {
            try {
                // Load from localStorage
                const savedFactSheet = localStorage.getItem('datasafe_fact_sheet');
                const savedStrategies = localStorage.getItem('datasafe_strategies');
                
                if (savedFactSheet) {
                    companyInfoPlaceholder.style.display = 'none';
                    companyInfoContent.style.display = 'block';
                    companyInfoContent.innerHTML = `
                        <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6;">
                            <div style="white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; margin: 0; font-size: 0.875rem; line-height: 1.5;">${savedFactSheet}</div>
                        </div>
                    `;
                    document.getElementById('download-company-btn').style.display = 'inline-block';
                    document.getElementById('save-company-btn').style.display = 'inline-block';
                }
                
                if (savedStrategies) {
                    strategiesPlaceholder.style.display = 'none';
                    strategiesContent.style.display = 'block';
                    strategiesContent.innerHTML = `
                        <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; border-left: 4px solid #f59e0b;">
                            <div style="white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; margin: 0; font-size: 0.875rem; line-height: 1.5;">${savedStrategies}</div>
                        </div>
                    `;
                    document.getElementById('download-strategies-btn').style.display = 'inline-block';
                    document.getElementById('save-strategies-btn').style.display = 'inline-block';
                }
                
                // Load from server as backup
                const res = await fetch('/api/user_chats');
                const data = await res.json();
                
                if (data && data.length > 0) {
                    const latestChat = data[0];
                    
                    if (!savedFactSheet && latestChat.fact_sheet) {
                        companyInfoPlaceholder.style.display = 'none';
                        companyInfoContent.style.display = 'block';
                        companyInfoContent.innerHTML = `
                            <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6;">
                                <div style="white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; margin: 0; font-size: 0.875rem; line-height: 1.5;">${latestChat.fact_sheet}</div>
                            </div>
                        `;
                        document.getElementById('download-company-btn').style.display = 'inline-block';
                        document.getElementById('save-company-btn').style.display = 'inline-block';
                        localStorage.setItem('datasafe_fact_sheet', latestChat.fact_sheet);
                    }
                    
                    if (!savedStrategies && latestChat.strategies) {
                        strategiesPlaceholder.style.display = 'none';
                        strategiesContent.style.display = 'block';
                        strategiesContent.innerHTML = `
                            <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; border-left: 4px solid #f59e0b;">
                                <div style="white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; margin: 0; font-size: 0.875rem; line-height: 1.5;">${latestChat.strategies}</div>
                            </div>
                        `;
                        document.getElementById('download-strategies-btn').style.display = 'inline-block';
                        document.getElementById('save-strategies-btn').style.display = 'inline-block';
                        localStorage.setItem('datasafe_strategies', latestChat.strategies);
                    }
                }
            } catch (error) {
                console.error('Error loading saved data:', error);
            }
        });
    </script>
</body>
</html>
EOF

echo "📤 Uploading updated template to Lightsail instance..."
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no /tmp/user_dashboard_updated.html ubuntu@$INSTANCE_IP:/tmp/

echo "🔧 Updating template on server..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP << 'EOF'
    echo "📝 Updating user dashboard template..."
    sudo cp /tmp/user_dashboard_updated.html /opt/datasafe/backend/templates/user_dashboard.html
    sudo chown ubuntu:ubuntu /opt/datasafe/backend/templates/user_dashboard.html
    sudo chmod 644 /opt/datasafe/backend/templates/user_dashboard.html
    
    echo "🔄 Restarting application..."
    cd /opt/datasafe
    docker-compose restart
    
    echo "⏳ Waiting for application to restart..."
    sleep 10
    
    echo "✅ Template updated successfully!"
    echo "🌐 Your updated design is now live at: http://$INSTANCE_IP"
EOF

echo ""
echo "🎉 Design update complete!"
echo "🌐 Visit your Lightsail instance to see the new design:"
echo "   http://$INSTANCE_IP"
echo ""
echo "✨ The new design features:"
echo "   • Clean, modern layout"
echo "   • Better organized sidebar"
echo "   • Responsive design for all devices"
echo "   • Improved user experience"
echo "   • Less cluttered interface"

# Clean up
rm -f /tmp/user_dashboard_updated.html 