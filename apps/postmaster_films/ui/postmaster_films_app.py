"""Postmaster Films - Streamlit Operations Console"""

import streamlit as st
import requests
import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Configuration
API_BASE = os.environ.get("POSTMASTER_API", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Postmaster Films - Ops Console",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .status-success { color: #10b981; font-weight: bold; }
    .status-error { color: #ef4444; font-weight: bold; }
    .status-processing { color: #f59e0b; font-weight: bold; }
    .cost-display { 
        background-color: #f0f9ff; 
        padding: 0.5rem; 
        border-radius: 5px; 
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

def api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API request with error handling"""
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return {}

def format_status(status: str) -> str:
    """Format status with color coding"""
    status_colors = {
        "COMPLETE": "status-success",
        "DRAFT": "status-processing", 
        "RENDERING": "status-processing",
        "ASSEMBLING": "status-processing",
        "ERROR": "status-error",
        "FAILED": "status-error"
    }
    color_class = status_colors.get(status, "")
    return f'<span class="{color_class}">{status}</span>'

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 Postmaster Films - AI TV Studio</h1>
        <p>End-to-end video production pipeline with budget management</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Dashboard",
        "Create Episode", 
        "Manage Episodes",
        "Scene Management",
        "Job Monitor",
        "Asset Library",
        "System Status"
    ])
    
    # Route to appropriate page
    if page == "Dashboard":
        show_dashboard()
    elif page == "Create Episode":
        show_create_episode()
    elif page == "Manage Episodes":
        show_manage_episodes()
    elif page == "Scene Management":
        show_scene_management()
    elif page == "Job Monitor":
        show_job_monitor()
    elif page == "Asset Library":
        show_asset_library()
    elif page == "System Status":
        show_system_status()

def show_dashboard():
    """Show main dashboard"""
    st.header("📊 Production Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get stats
    projects = api_request("/projects/")
    episodes = api_request("/episodes/")
    jobs = api_request("/jobs/")
    
    with col1:
        st.metric("Projects", len(projects))
    
    with col2:
        st.metric("Episodes", len(episodes))
    
    with col3:
        active_jobs = len([j for j in jobs if j.get("status") in ["PENDING", "PROCESSING"]])
        st.metric("Active Jobs", active_jobs)
    
    with col4:
        completed_episodes = len([e for e in episodes if e.get("status") == "COMPLETE"])
        st.metric("Completed Episodes", completed_episodes)
    
    # Recent activity
    st.subheader("📈 Recent Activity")
    
    if episodes:
        df_episodes = pd.DataFrame(episodes)
        df_episodes["created_at"] = pd.to_datetime(df_episodes["created_at"])
        df_episodes = df_episodes.sort_values("created_at", ascending=False).head(10)
        
        st.dataframe(
            df_episodes[["id", "title", "status", "budget_usd", "veo_spend_usd", "created_at"]],
            use_container_width=True
        )

def show_create_episode():
    """Show episode creation interface"""
    st.header("🎬 Create New Episode")
    
    # Get projects for dropdown
    projects = api_request("/projects/")
    
    if not projects:
        st.warning("No projects found. Create a project first.")
        with st.expander("Create New Project"):
            proj_name = st.text_input("Project Name")
            proj_client = st.text_input("Client Name")
            proj_notes = st.text_area("Notes")
            
            if st.button("Create Project"):
                project_data = {
                    "name": proj_name,
                    "client": proj_client,
                    "notes": proj_notes
                }
                result = api_request("/projects/", "POST", project_data)
                if result:
                    st.success("Project created successfully!")
                    st.rerun()
        return
    
    # Episode creation form
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Episode Details")
        
        project_options = {p["name"]: p["id"] for p in projects}
        selected_project = st.selectbox("Project", list(project_options.keys()))
        project_id = project_options[selected_project]
        
        title = st.text_input("Episode Title", placeholder="e.g., AI News Update #1")
        budget = st.number_input("Budget (USD)", min_value=0.0, value=50.0, step=5.0)
        
        script_text = st.text_area(
            "Script Content", 
            height=300,
            placeholder="Paste your script here. Each paragraph will become a scene."
        )
    
    with col2:
        st.subheader("💰 Budget Calculator")
        
        if budget > 0:
            veo_seconds = int(budget / 0.40)  # $0.40 per second for Veo
            st.markdown(f"""
            <div class="cost-display">
                <strong>Budget Breakdown:</strong><br>
                Total Budget: ${budget}<br>
                Max Veo Seconds: {veo_seconds}s<br>
                Price per Second: $0.40<br>
                <small>Remaining scenes use free AnimateDiff</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("📋 Scene Preview")
        if script_text:
            paragraphs = [p.strip() for p in script_text.split("\n\n") if p.strip()]
            st.info(f"Will create {len(paragraphs)} scenes")
            for i, para in enumerate(paragraphs[:3]):
                st.text(f"Scene {i+1}: {para[:50]}...")
            if len(paragraphs) > 3:
                st.text(f"... and {len(paragraphs) - 3} more scenes")
    
    # Create episode button
    if st.button("🚀 Create Episode from Script", type="primary"):
        if not title or not script_text:
            st.error("Please provide both title and script content")
            return
        
        episode_data = {
            "project_id": project_id,
            "title": title,
            "budget_usd": budget,
            "script_text": script_text
        }
        
        with st.spinner("Creating episode and generating scenes..."):
            result = api_request("/episodes/from_script", "POST", episode_data)
        
        if result:
            st.success(f"✅ Episode created with ID: {result['id']}")
            st.success(f"Generated {len(result.get('scenes', []))} scenes")
            st.balloons()
        else:
            st.error("Failed to create episode")

def show_manage_episodes():
    """Show episode management interface"""
    st.header("📝 Manage Episodes")
    
    episodes = api_request("/episodes/")
    
    if not episodes:
        st.info("No episodes found. Create one first!")
        return
    
    # Episode selection
    episode_options = {f"{e['title']} (ID: {e['id']})": e for e in episodes}
    selected_episode_key = st.selectbox("Select Episode", list(episode_options.keys()))
    episode = episode_options[selected_episode_key]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Status", episode["status"])
        st.metric("Budget", f"${episode['budget_usd']}")
    
    with col2:
        st.metric("Veo Spend", f"${episode['veo_spend_usd']}")
        remaining = episode['budget_usd'] - episode['veo_spend_usd']
        st.metric("Remaining", f"${remaining}")
    
    with col3:
        scenes_count = len(episode.get('scenes', []))
        st.metric("Scenes", scenes_count)
    
    # Action buttons
    st.subheader("🎬 Production Actions")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🎨 Render All Scenes", type="primary"):
            with st.spinner("Queuing render job..."):
                result = api_request(f"/jobs/render_episode/{episode['id']}", "POST")
            if result:
                st.success(f"Render job queued! Job ID: {result['job_id']}")
    
    with action_col2:
        if st.button("🔗 Assemble Episode"):
            with st.spinner("Queuing assembly job..."):
                result = api_request(f"/jobs/assemble/{episode['id']}", "POST")
            if result:
                st.success(f"Assembly job queued! Job ID: {result['job_id']}")
    
    with action_col3:
        if st.button("🎤 Add Voiceover"):
            with st.spinner("Queuing voiceover job..."):
                result = api_request(f"/jobs/mux_vo/{episode['id']}", "POST")
            if result:
                st.success(f"Voiceover job queued! Job ID: {result['job_id']}")
    
    # Scene details
    if episode.get('scenes'):
        st.subheader("🎭 Scene Details")
        scenes_df = pd.DataFrame(episode['scenes'])
        st.dataframe(scenes_df, use_container_width=True)

def show_scene_management():
    """Show scene editing interface"""
    st.header("🎭 Scene Management")
    
    episodes = api_request("/episodes/")
    if not episodes:
        st.info("No episodes found.")
        return
    
    # Episode selection
    episode_options = {f"{e['title']} (ID: {e['id']})": e for e in episodes}
    selected_episode_key = st.selectbox("Select Episode", list(episode_options.keys()))
    episode = episode_options[selected_episode_key]
    
    scenes = episode.get('scenes', [])
    if not scenes:
        st.info("No scenes found in this episode.")
        return
    
    # Scene selection
    scene_options = {f"Scene {s['index']}: {s['description'][:50]}...": s for s in scenes}
    selected_scene_key = st.selectbox("Select Scene", list(scene_options.keys()))
    scene = scene_options[selected_scene_key]
    
    # Scene editing form
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Edit Scene")
        
        new_description = st.text_area(
            "Description", 
            value=scene['description'],
            height=100
        )
        
        new_duration = st.number_input(
            "Duration (seconds)", 
            min_value=1, 
            max_value=30, 
            value=scene['duration_sec']
        )
        
        new_scene_type = st.selectbox(
            "Scene Type",
            ["FILLER", "HERO"],
            index=0 if scene['scene_type'] == "FILLER" else 1
        )
        
        new_prompt = st.text_area(
            "Custom Prompt (optional)",
            value=scene.get('prompt', ''),
            height=100
        )
    
    with col2:
        st.subheader("Scene Info")
        st.info(f"**Index:** {scene['index']}")
        st.info(f"**Type:** {scene['scene_type']}")
        st.info(f"**Model Route:** {scene.get('model_route', 'Not set')}")
        st.info(f"**Cost:** ${scene.get('cost_usd', 0)}")
        
        if scene.get('output_video_path'):
            st.success("✅ Scene rendered")
        else:
            st.warning("⏳ Not rendered yet")
    
    if st.button("💾 Update Scene"):
        update_data = {
            "description": new_description,
            "duration_sec": new_duration,
            "scene_type": new_scene_type,
            "prompt": new_prompt if new_prompt else None
        }
        
        result = api_request(f"/scenes/{scene['id']}", "PUT", update_data)
        if result:
            st.success("Scene updated successfully!")
            st.rerun()

def show_job_monitor():
    """Show job monitoring interface"""
    st.header("⚡ Job Monitor")
    
    jobs = api_request("/jobs/")
    
    if not jobs:
        st.info("No jobs found.")
        return
    
    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox(
            "Filter by Status", 
            ["All", "PENDING", "PROCESSING", "COMPLETE", "FAILED"]
        )
    with col2:
        kind_filter = st.selectbox(
            "Filter by Type",
            ["All", "render_episode", "assemble_episode", "voiceover_mux"]
        )
    
    # Filter jobs
    filtered_jobs = jobs
    if status_filter != "All":
        filtered_jobs = [j for j in filtered_jobs if j.get("status") == status_filter]
    if kind_filter != "All":
        filtered_jobs = [j for j in filtered_jobs if j.get("kind") == kind_filter]
    
    # Jobs table
    if filtered_jobs:
        jobs_data = []
        for job in filtered_jobs:
            jobs_data.append({
                "ID": job["id"],
                "Type": job["kind"],
                "Status": job["status"],
                "Created": job["created_at"][:19],
                "Started": job.get("started_at", "")[:19] if job.get("started_at") else "-",
                "Completed": job.get("completed_at", "")[:19] if job.get("completed_at") else "-"
            })
        
        jobs_df = pd.DataFrame(jobs_data)
        st.dataframe(jobs_df, use_container_width=True)
        
        # Job details
        if st.checkbox("Show Job Details"):
            selected_job_id = st.selectbox("Select Job ID", [j["id"] for j in filtered_jobs])
            selected_job = next(j for j in filtered_jobs if j["id"] == selected_job_id)
            
            st.json(selected_job)
    else:
        st.info("No jobs match the current filters.")

def show_asset_library():
    """Show asset management interface"""
    st.header("📁 Asset Library")
    
    # Asset library stats
    stats = api_request("/assets/library/stats")
    if stats:
        library_stats = stats.get("library_stats", {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Assets", library_stats.get("total_assets", 0))
        with col2:
            st.metric("Storage (MB)", library_stats.get("total_size_mb", 0))
        with col3:
            st.metric("Reference Frames", library_stats.get("ref_frame", 0))
        with col4:
            st.metric("Backgrounds", library_stats.get("background", 0))
    
    # Asset upload
    st.subheader("📤 Upload New Asset")
    
    col1, col2 = st.columns(2)
    with col1:
        asset_kind = st.selectbox("Asset Type", [
            "ref_frame", "background", "style", "template", "audio", "general"
        ])
        asset_label = st.text_input("Asset Label")
    
    with col2:
        uploaded_file = st.file_uploader("Choose file", type=["jpg", "png", "mp4", "wav", "json"])
    
    if uploaded_file and asset_label and st.button("Upload Asset"):
        # Note: Streamlit file upload to FastAPI requires multipart/form-data
        # This is a simplified example - in production you'd handle the upload properly
        st.info("File upload functionality would be implemented with proper multipart handling")
    
    # Asset list
    assets = api_request("/assets/")
    if assets:
        st.subheader("📚 Asset Library")
        
        assets_data = []
        for asset in assets:
            assets_data.append({
                "ID": asset["id"],
                "Kind": asset["kind"],
                "Label": asset["label"],
                "Path": asset["path"],
                "Created": asset["created_at"][:19]
            })
        
        assets_df = pd.DataFrame(assets_data)
        st.dataframe(assets_df, use_container_width=True)

def show_system_status():
    """Show system status and configuration"""
    st.header("🔧 System Status")
    
    # Health check
    health = api_request("/health")
    config = api_request("/config")
    
    if health:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏥 Health Status")
            status_color = "🟢" if health.get("status") == "healthy" else "🔴"
            st.write(f"{status_color} System Status: {health.get('status', 'unknown')}")
            st.write(f"📂 Media Root: {health.get('media_root', 'unknown')}")
            st.write(f"🎥 Veo Enabled: {'✅' if health.get('veo_enabled') else '❌'}")
            st.write(f"🎤 TTS Enabled: {'✅' if health.get('tts_enabled') else '❌'}")
        
        with col2:
            st.subheader("⚙️ Configuration")
            if config:
                st.json(config)
    
    # API endpoints test
    st.subheader("🔌 API Endpoints Test")
    
    endpoints = [
        "/health",
        "/config", 
        "/projects/",
        "/episodes/",
        "/jobs/"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{API_BASE}{endpoint}")
            status = "🟢" if response.status_code == 200 else "🔴"
            st.write(f"{status} {endpoint} - Status: {response.status_code}")
        except Exception as e:
            st.write(f"🔴 {endpoint} - Error: {str(e)}")

if __name__ == "__main__":
    main()

