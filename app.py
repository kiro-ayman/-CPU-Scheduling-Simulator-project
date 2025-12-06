import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import copy
import io
import time

st.set_page_config(
    page_title="CPU Scheduling Simulator",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    .state-box {
        display: inline-block;
        padding: 10px 20px;
        margin: 5px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
    .state-new { background-color: #e3f2fd; color: #1565c0; border: 2px solid #1565c0; }
    .state-ready { background-color: #fff3e0; color: #ef6c00; border: 2px solid #ef6c00; }
    .state-running { background-color: #e8f5e9; color: #2e7d32; border: 2px solid #2e7d32; }
    .state-waiting { background-color: #fce4ec; color: #c2185b; border: 2px solid #c2185b; }
    .state-terminated { background-color: #f3e5f5; color: #7b1fa2; border: 2px solid #7b1fa2; }
</style>
""", unsafe_allow_html=True)


@dataclass
class Process:
    pid: str
    arrival_time: int
    burst_time: int
    priority: int = 0
    queue_level: int = 1
    remaining_time: int = 0
    completion_time: int = 0
    waiting_time: int = 0
    turnaround_time: int = 0
    response_time: int = -1
    state_history: List[Tuple[int, str]] = field(default_factory=list)

    def __post_init__(self):
        self.remaining_time = self.burst_time
        if not self.state_history:
            self.state_history = [(self.arrival_time, "New")]


class CPUScheduler:
    def __init__(self, processes: List[Process]):
        self.processes = [copy.deepcopy(p) for p in processes]
        self.gantt_chart = []
        self.time = 0
        self.execution_steps = []

    def _record_step(self, time: int, running: str, ready_queue: List[str], 
                     completed: List[str], description: str):
        self.execution_steps.append({
            "time": time,
            "running": running,
            "ready_queue": ready_queue.copy(),
            "completed": completed.copy(),
            "description": description
        })

    def fcfs(self) -> Tuple[List[Process], List[dict], List[dict]]:
        processes = sorted(self.processes, key=lambda x: (x.arrival_time, x.pid))
        gantt = []
        current_time = 0
        steps = []
        completed_list = []

        for p in processes:
            ready_queue = [proc.pid for proc in processes 
                          if proc.arrival_time <= current_time and proc.pid != p.pid 
                          and proc.pid not in completed_list]
            
            if current_time < p.arrival_time:
                steps.append({
                    "time": current_time,
                    "running": "Idle",
                    "ready_queue": [],
                    "completed": completed_list.copy(),
                    "description": f"CPU idle, waiting for {p.pid} to arrive at t={p.arrival_time}"
                })
                gantt.append({"Process": "Idle", "Start": current_time, "End": p.arrival_time})
                current_time = p.arrival_time

            p.state_history.append((current_time, "Ready"))
            p.state_history.append((current_time, "Running"))
            p.response_time = current_time - p.arrival_time
            start = current_time
            
            steps.append({
                "time": current_time,
                "running": p.pid,
                "ready_queue": ready_queue,
                "completed": completed_list.copy(),
                "description": f"{p.pid} starts execution (burst time: {p.burst_time})"
            })
            
            current_time += p.burst_time
            p.completion_time = current_time
            p.turnaround_time = p.completion_time - p.arrival_time
            p.waiting_time = p.turnaround_time - p.burst_time
            p.state_history.append((current_time, "Terminated"))
            completed_list.append(p.pid)

            gantt.append({"Process": p.pid, "Start": start, "End": current_time})
            
            steps.append({
                "time": current_time,
                "running": "None",
                "ready_queue": [proc.pid for proc in processes 
                               if proc.arrival_time <= current_time and proc.pid not in completed_list],
                "completed": completed_list.copy(),
                "description": f"{p.pid} completed at t={current_time}"
            })

        return processes, gantt, steps

    def sjf(self) -> Tuple[List[Process], List[dict], List[dict]]:
        processes = [copy.deepcopy(p) for p in self.processes]
        n = len(processes)
        completed = 0
        current_time = 0
        gantt = []
        steps = []
        is_completed = [False] * n
        completed_list = []

        while completed < n:
            ready = []
            for i, p in enumerate(processes):
                if p.arrival_time <= current_time and not is_completed[i]:
                    ready.append((i, p))

            if not ready:
                next_arrival = min(p.arrival_time for i, p in enumerate(processes) if not is_completed[i])
                steps.append({
                    "time": current_time,
                    "running": "Idle",
                    "ready_queue": [],
                    "completed": completed_list.copy(),
                    "description": f"CPU idle until t={next_arrival}"
                })
                gantt.append({"Process": "Idle", "Start": current_time, "End": next_arrival})
                current_time = next_arrival
                continue

            idx, process = min(ready, key=lambda x: (x[1].burst_time, x[1].arrival_time))
            
            ready_queue = [p.pid for _, p in ready if p.pid != process.pid]
            
            process.state_history.append((current_time, "Ready"))
            process.state_history.append((current_time, "Running"))
            process.response_time = current_time - process.arrival_time
            start = current_time
            
            steps.append({
                "time": current_time,
                "running": process.pid,
                "ready_queue": ready_queue,
                "completed": completed_list.copy(),
                "description": f"{process.pid} selected (shortest burst: {process.burst_time})"
            })
            
            current_time += process.burst_time
            process.completion_time = current_time
            process.turnaround_time = process.completion_time - process.arrival_time
            process.waiting_time = process.turnaround_time - process.burst_time
            process.state_history.append((current_time, "Terminated"))
            completed_list.append(process.pid)

            gantt.append({"Process": process.pid, "Start": start, "End": current_time})
            is_completed[idx] = True
            completed += 1

        return processes, gantt, steps

    def srtf(self) -> Tuple[List[Process], List[dict], List[dict]]:
        processes = [copy.deepcopy(p) for p in self.processes]
        n = len(processes)
        current_time = 0
        completed = 0
        gantt = []
        steps = []
        prev_process = None
        prev_start = 0
        completed_list = []

        while completed < n:
            ready = [p for p in processes if p.arrival_time <= current_time and p.remaining_time > 0]

            if not ready:
                remaining = [p for p in processes if p.remaining_time > 0]
                if remaining:
                    next_time = min(p.arrival_time for p in remaining)
                    if prev_process and prev_process != "Idle":
                        gantt.append({"Process": prev_process, "Start": prev_start, "End": current_time})
                    gantt.append({"Process": "Idle", "Start": current_time, "End": next_time})
                    current_time = next_time
                    prev_process = None
                continue

            current = min(ready, key=lambda x: (x.remaining_time, x.arrival_time))

            if prev_process != current.pid:
                if prev_process is not None and prev_process != "Idle":
                    gantt.append({"Process": prev_process, "Start": prev_start, "End": current_time})
                prev_process = current.pid
                prev_start = current_time
                
                ready_queue = [p.pid for p in ready if p.pid != current.pid]
                steps.append({
                    "time": current_time,
                    "running": current.pid,
                    "ready_queue": ready_queue,
                    "completed": completed_list.copy(),
                    "description": f"{current.pid} running (remaining: {current.remaining_time})"
                })

            if current.response_time == -1:
                current.response_time = current_time - current.arrival_time
                current.state_history.append((current_time, "Running"))

            current.remaining_time -= 1
            current_time += 1

            if current.remaining_time == 0:
                current.completion_time = current_time
                current.turnaround_time = current.completion_time - current.arrival_time
                current.waiting_time = current.turnaround_time - current.burst_time
                current.state_history.append((current_time, "Terminated"))
                completed_list.append(current.pid)
                completed += 1

        if prev_process and prev_process != "Idle":
            gantt.append({"Process": prev_process, "Start": prev_start, "End": current_time})

        return processes, gantt, steps

    def round_robin(self, quantum: int) -> Tuple[List[Process], List[dict], List[dict]]:
        processes = [copy.deepcopy(p) for p in self.processes]
        n = len(processes)
        queue = []
        current_time = 0
        gantt = []
        steps = []
        completed = 0
        arrived = [False] * n
        completed_list = []

        sorted_indices = sorted(range(n), key=lambda i: processes[i].arrival_time)

        while completed < n:
            for i in sorted_indices:
                if not arrived[i] and processes[i].arrival_time <= current_time:
                    queue.append(i)
                    arrived[i] = True
                    processes[i].state_history.append((current_time, "Ready"))

            if not queue:
                remaining = [i for i in range(n) if processes[i].remaining_time > 0]
                if remaining:
                    next_time = min(processes[i].arrival_time for i in remaining if not arrived[i])
                    gantt.append({"Process": "Idle", "Start": current_time, "End": next_time})
                    current_time = next_time
                    continue
                break

            idx = queue.pop(0)
            p = processes[idx]

            if p.response_time == -1:
                p.response_time = current_time - p.arrival_time

            p.state_history.append((current_time, "Running"))
            exec_time = min(quantum, p.remaining_time)
            start = current_time
            
            ready_queue = [processes[i].pid for i in queue]
            steps.append({
                "time": current_time,
                "running": p.pid,
                "ready_queue": ready_queue,
                "completed": completed_list.copy(),
                "description": f"{p.pid} executing for {exec_time} units (Q={quantum})"
            })
            
            current_time += exec_time
            p.remaining_time -= exec_time

            gantt.append({"Process": p.pid, "Start": start, "End": current_time})

            for i in sorted_indices:
                if not arrived[i] and processes[i].arrival_time <= current_time:
                    queue.append(i)
                    arrived[i] = True
                    processes[i].state_history.append((current_time, "Ready"))

            if p.remaining_time > 0:
                queue.append(idx)
                p.state_history.append((current_time, "Ready"))
            else:
                p.completion_time = current_time
                p.turnaround_time = p.completion_time - p.arrival_time
                p.waiting_time = p.turnaround_time - p.burst_time
                p.state_history.append((current_time, "Terminated"))
                completed_list.append(p.pid)
                completed += 1

        return processes, gantt, steps

    def priority_scheduling(self, preemptive: bool = False) -> Tuple[List[Process], List[dict], List[dict]]:
        processes = [copy.deepcopy(p) for p in self.processes]
        n = len(processes)
        current_time = 0
        completed = 0
        gantt = []
        steps = []
        is_completed = [False] * n
        completed_list = []

        if preemptive:
            prev_process = None
            prev_start = 0

            while completed < n:
                ready = [(i, p) for i, p in enumerate(processes) 
                         if p.arrival_time <= current_time and p.remaining_time > 0]

                if not ready:
                    remaining = [p for p in processes if p.remaining_time > 0]
                    if remaining:
                        next_time = min(p.arrival_time for p in remaining)
                        if prev_process:
                            gantt.append({"Process": prev_process, "Start": prev_start, "End": current_time})
                        gantt.append({"Process": "Idle", "Start": current_time, "End": next_time})
                        current_time = next_time
                        prev_process = None
                    continue

                idx, current = min(ready, key=lambda x: (x[1].priority, x[1].arrival_time))

                if prev_process != current.pid:
                    if prev_process is not None:
                        gantt.append({"Process": prev_process, "Start": prev_start, "End": current_time})
                    prev_process = current.pid
                    prev_start = current_time
                    
                    ready_queue = [p.pid for _, p in ready if p.pid != current.pid]
                    steps.append({
                        "time": current_time,
                        "running": current.pid,
                        "ready_queue": ready_queue,
                        "completed": completed_list.copy(),
                        "description": f"{current.pid} running (priority: {current.priority})"
                    })

                if current.response_time == -1:
                    current.response_time = current_time - current.arrival_time
                    current.state_history.append((current_time, "Running"))

                current.remaining_time -= 1
                current_time += 1

                if current.remaining_time == 0:
                    current.completion_time = current_time
                    current.turnaround_time = current.completion_time - current.arrival_time
                    current.waiting_time = current.turnaround_time - current.burst_time
                    current.state_history.append((current_time, "Terminated"))
                    completed_list.append(current.pid)
                    completed += 1

            if prev_process:
                gantt.append({"Process": prev_process, "Start": prev_start, "End": current_time})
        else:
            while completed < n:
                ready = [(i, p) for i, p in enumerate(processes) 
                         if p.arrival_time <= current_time and not is_completed[i]]

                if not ready:
                    next_arrival = min(p.arrival_time for i, p in enumerate(processes) if not is_completed[i])
                    gantt.append({"Process": "Idle", "Start": current_time, "End": next_arrival})
                    current_time = next_arrival
                    continue

                idx, process = min(ready, key=lambda x: (x[1].priority, x[1].arrival_time))
                
                ready_queue = [p.pid for _, p in ready if p.pid != process.pid]
                
                process.state_history.append((current_time, "Ready"))
                process.state_history.append((current_time, "Running"))
                process.response_time = current_time - process.arrival_time
                start = current_time
                
                steps.append({
                    "time": current_time,
                    "running": process.pid,
                    "ready_queue": ready_queue,
                    "completed": completed_list.copy(),
                    "description": f"{process.pid} selected (priority: {process.priority})"
                })
                
                current_time += process.burst_time
                process.completion_time = current_time
                process.turnaround_time = process.completion_time - process.arrival_time
                process.waiting_time = process.turnaround_time - process.burst_time
                process.state_history.append((current_time, "Terminated"))
                completed_list.append(process.pid)

                gantt.append({"Process": process.pid, "Start": start, "End": current_time})
                is_completed[idx] = True
                completed += 1

        return processes, gantt, steps

    def multilevel_queue(self, queue_quantums: Dict[int, int]) -> Tuple[List[Process], List[dict], List[dict]]:
        processes = [copy.deepcopy(p) for p in self.processes]
        n = len(processes)
        current_time = 0
        gantt = []
        steps = []
        completed = 0
        completed_list = []
        arrived = [False] * n
        
        queues = {level: [] for level in queue_quantums.keys()}
        
        sorted_indices = sorted(range(n), key=lambda i: processes[i].arrival_time)

        while completed < n:
            for i in sorted_indices:
                if not arrived[i] and processes[i].arrival_time <= current_time:
                    level = processes[i].queue_level
                    if level in queues:
                        queues[level].append(i)
                    else:
                        queues[min(queues.keys())].append(i)
                    arrived[i] = True
                    processes[i].state_history.append((current_time, "Ready"))

            selected_idx = None
            selected_level = None
            
            for level in sorted(queues.keys()):
                if queues[level]:
                    selected_idx = queues[level].pop(0)
                    selected_level = level
                    break

            if selected_idx is None:
                remaining = [i for i in range(n) if processes[i].remaining_time > 0]
                if remaining:
                    not_arrived = [i for i in remaining if not arrived[i]]
                    if not_arrived:
                        next_time = min(processes[i].arrival_time for i in not_arrived)
                        gantt.append({"Process": "Idle", "Start": current_time, "End": next_time})
                        current_time = next_time
                        continue
                break

            p = processes[selected_idx]
            quantum = queue_quantums.get(selected_level, 2)

            if p.response_time == -1:
                p.response_time = current_time - p.arrival_time

            p.state_history.append((current_time, "Running"))
            exec_time = min(quantum, p.remaining_time)
            start = current_time
            
            all_ready = []
            for level, q in queues.items():
                all_ready.extend([processes[i].pid for i in q])
            
            steps.append({
                "time": current_time,
                "running": p.pid,
                "ready_queue": all_ready,
                "completed": completed_list.copy(),
                "description": f"{p.pid} from Queue {selected_level} (Q={quantum})"
            })
            
            current_time += exec_time
            p.remaining_time -= exec_time

            gantt.append({"Process": p.pid, "Start": start, "End": current_time})

            for i in sorted_indices:
                if not arrived[i] and processes[i].arrival_time <= current_time:
                    level = processes[i].queue_level
                    if level in queues:
                        queues[level].append(i)
                    else:
                        queues[min(queues.keys())].append(i)
                    arrived[i] = True
                    processes[i].state_history.append((current_time, "Ready"))

            if p.remaining_time > 0:
                queues[selected_level].append(selected_idx)
                p.state_history.append((current_time, "Ready"))
            else:
                p.completion_time = current_time
                p.turnaround_time = p.completion_time - p.arrival_time
                p.waiting_time = p.turnaround_time - p.burst_time
                p.state_history.append((current_time, "Terminated"))
                completed_list.append(p.pid)
                completed += 1

        return processes, gantt, steps


def create_gantt_chart(gantt_data: List[dict], title: str = "CPU Scheduling Gantt Chart"):
    if not gantt_data:
        return None

    df = pd.DataFrame(gantt_data)
    
    colors = {
        "Idle": "#e0e0e0",
        "P1": "#667eea",
        "P2": "#764ba2",
        "P3": "#f093fb",
        "P4": "#f5576c",
        "P5": "#4facfe",
        "P6": "#00f2fe",
        "P7": "#43e97b",
        "P8": "#fa709a",
        "P9": "#fee140",
        "P10": "#30cfd0"
    }

    fig = go.Figure()

    for _, row in df.iterrows():
        color = colors.get(row["Process"], "#667eea")
        fig.add_trace(go.Bar(
            x=[row["End"] - row["Start"]],
            y=[0],
            orientation='h',
            base=row["Start"],
            name=row["Process"],
            marker_color=color,
            text=row["Process"],
            textposition='inside',
            textfont=dict(color='white', size=12, family='Arial Black'),
            hovertemplate=f"<b>{row['Process']}</b><br>Start: {row['Start']}<br>End: {row['End']}<br>Duration: {row['End']-row['Start']}<extra></extra>",
            showlegend=False
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#333')),
        barmode='stack',
        xaxis=dict(
            title="Time Units",
            tickmode='linear',
            tick0=0,
            dtick=1,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(visible=False),
        height=200,
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig


def create_state_diagram(processes: List[Process]):
    fig = go.Figure()
    
    colors = {
        "New": "#1565c0",
        "Ready": "#ef6c00",
        "Running": "#2e7d32",
        "Waiting": "#c2185b",
        "Terminated": "#7b1fa2"
    }
    
    y_positions = {p.pid: i for i, p in enumerate(processes)}
    
    for p in processes:
        y = y_positions[p.pid]
        
        for i, (time, state) in enumerate(p.state_history):
            fig.add_trace(go.Scatter(
                x=[time],
                y=[y],
                mode='markers+text',
                marker=dict(size=20, color=colors.get(state, '#666')),
                text=[state[0]],
                textposition='middle center',
                textfont=dict(color='white', size=10),
                hovertemplate=f"<b>{p.pid}</b><br>Time: {time}<br>State: {state}<extra></extra>",
                showlegend=False
            ))
            
            if i > 0:
                prev_time, prev_state = p.state_history[i-1]
                fig.add_trace(go.Scatter(
                    x=[prev_time, time],
                    y=[y, y],
                    mode='lines',
                    line=dict(color='#999', width=2, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    for state, color in colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=color),
            name=state,
            showlegend=True
        ))
    
    fig.update_layout(
        title="Process State Transitions",
        xaxis=dict(title="Time", showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(
            tickmode='array',
            tickvals=list(y_positions.values()),
            ticktext=list(y_positions.keys()),
            title="Process"
        ),
        height=max(300, len(processes) * 60),
        margin=dict(l=60, r=20, t=60, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return fig


def calculate_metrics(processes: List[Process]) -> dict:
    n = len(processes)
    if n == 0:
        return {}

    total_burst = sum(p.burst_time for p in processes)
    completion_time = max(p.completion_time for p in processes)
    
    return {
        "avg_waiting_time": sum(p.waiting_time for p in processes) / n,
        "avg_turnaround_time": sum(p.turnaround_time for p in processes) / n,
        "avg_response_time": sum(p.response_time for p in processes) / n,
        "cpu_utilization": (total_burst / completion_time * 100) if completion_time > 0 else 0,
        "throughput": n / completion_time if completion_time > 0 else 0
    }


def display_metrics(metrics: dict):
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Avg Waiting Time", f"{metrics.get('avg_waiting_time', 0):.2f}")
    with col2:
        st.metric("Avg Turnaround Time", f"{metrics.get('avg_turnaround_time', 0):.2f}")
    with col3:
        st.metric("Avg Response Time", f"{metrics.get('avg_response_time', 0):.2f}")
    with col4:
        st.metric("CPU Utilization", f"{metrics.get('cpu_utilization', 0):.1f}%")
    with col5:
        st.metric("Throughput", f"{metrics.get('throughput', 0):.3f}")


def display_step_by_step(steps: List[dict], gantt_data: List[dict]):
    if not steps:
        st.info("No execution steps available for this algorithm.")
        return
    
    st.subheader("Step-by-Step Execution")
    
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    with col1:
        if st.button("Previous", disabled=st.session_state.current_step <= 0):
            st.session_state.current_step -= 1
            st.rerun()
    
    with col2:
        if st.button("Next", disabled=st.session_state.current_step >= len(steps) - 1):
            st.session_state.current_step += 1
            st.rerun()
    
    with col3:
        if st.button("Reset"):
            st.session_state.current_step = 0
            st.rerun()
    
    with col4:
        st.session_state.current_step = st.slider(
            "Step", 0, len(steps) - 1, st.session_state.current_step
        )
    
    step = steps[st.session_state.current_step]
    
    st.markdown(f"### Time: {step['time']}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Running:**")
        if step['running'] and step['running'] != "None":
            st.success(step['running'])
        else:
            st.warning("Idle")
    
    with col2:
        st.markdown("**Ready Queue:**")
        if step['ready_queue']:
            st.info(" → ".join(step['ready_queue']))
        else:
            st.info("Empty")
    
    with col3:
        st.markdown("**Completed:**")
        if step['completed']:
            st.success(", ".join(step['completed']))
        else:
            st.info("None")
    
    st.markdown(f"**Action:** {step['description']}")
    
    current_gantt = [g for g in gantt_data if g['End'] <= step['time'] + 1]
    if current_gantt:
        fig = create_gantt_chart(current_gantt, "Execution Progress")
        st.plotly_chart(fig, use_container_width=True)


def export_to_csv(processes: List[Process], results: List[Process]) -> str:
    data = []
    for p in results:
        data.append({
            "Process": p.pid,
            "Arrival Time": p.arrival_time,
            "Burst Time": p.burst_time,
            "Priority": p.priority,
            "Queue Level": p.queue_level,
            "Completion Time": p.completion_time,
            "Turnaround Time": p.turnaround_time,
            "Waiting Time": p.waiting_time,
            "Response Time": p.response_time
        })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def parse_csv(uploaded_file) -> List[Process]:
    df = pd.read_csv(uploaded_file)
    processes = []
    
    required_cols = ['Arrival Time', 'Burst Time']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return []
    
    for i, row in df.iterrows():
        pid = row.get('Process', f'P{i+1}')
        processes.append(Process(
            pid=str(pid),
            arrival_time=int(row['Arrival Time']),
            burst_time=int(row['Burst Time']),
            priority=int(row.get('Priority', i+1)),
            queue_level=int(row.get('Queue Level', 1))
        ))
    
    return processes


def main():
    st.markdown('<h1 class="main-header">CPU Scheduling Simulator</h1>', unsafe_allow_html=True)
    st.markdown("---")

    if 'processes' not in st.session_state:
        st.session_state.processes = None

    with st.sidebar:
        st.header("Process Configuration")
        
        input_method = st.radio("Input Method", ["Manual Entry", "CSV Upload"])
        
        if input_method == "CSV Upload":
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file:
                st.session_state.processes = parse_csv(uploaded_file)
                if st.session_state.processes:
                    st.success(f"Loaded {len(st.session_state.processes)} processes")
            
            st.download_button(
                "Download Template CSV",
                "Process,Arrival Time,Burst Time,Priority,Queue Level\nP1,0,5,2,1\nP2,1,3,1,1\nP3,2,8,3,2\nP4,3,6,4,2",
                "template.csv",
                "text/csv"
            )
        else:
            num_processes = st.number_input("Number of Processes", min_value=1, max_value=10, value=4)
            
            st.subheader("Process Details")
            
            processes = []
            for i in range(num_processes):
                with st.expander(f"Process P{i+1}", expanded=(i == 0)):
                    col1, col2 = st.columns(2)
                    with col1:
                        arrival = st.number_input(f"Arrival", min_value=0, value=i, key=f"arr_{i}")
                        burst = st.number_input(f"Burst", min_value=1, value=max(1, 5-i), key=f"burst_{i}")
                    with col2:
                        priority = st.number_input(f"Priority", min_value=1, value=i+1, key=f"prio_{i}")
                        queue_level = st.number_input(f"Queue Level", min_value=1, max_value=3, value=((i % 3) + 1), key=f"queue_{i}")
                    
                    processes.append(Process(
                        pid=f"P{i+1}",
                        arrival_time=arrival,
                        burst_time=burst,
                        priority=priority,
                        queue_level=queue_level
                    ))
            
            st.session_state.processes = processes
        
        st.markdown("---")
        st.subheader("Algorithm Settings")
        
        algorithm = st.selectbox(
            "Select Algorithm",
            ["FCFS (First Come First Serve)", 
             "SJF (Shortest Job First)", 
             "SRTF (Shortest Remaining Time First)",
             "Round Robin", 
             "Priority (Non-Preemptive)",
             "Priority (Preemptive)",
             "Multilevel Queue",
             "Compare All",
             "Preemptive vs Non-Preemptive"]
        )
        
        quantum = 2
        if "Round Robin" in algorithm or "Multilevel" in algorithm:
            quantum = st.number_input("Time Quantum", min_value=1, max_value=10, value=2)
        
        if "Multilevel" in algorithm:
            st.markdown("**Queue Time Quantums:**")
            q1 = st.number_input("Queue 1 (High Priority)", min_value=1, value=2, key="q1")
            q2 = st.number_input("Queue 2 (Medium Priority)", min_value=1, value=4, key="q2")
            q3 = st.number_input("Queue 3 (Low Priority)", min_value=1, value=8, key="q3")
            queue_quantums = {1: q1, 2: q2, 3: q3}
        else:
            queue_quantums = {1: 2, 2: 4, 3: 8}
        
        show_steps = st.checkbox("Show Step-by-Step Execution", value=False)
        show_state_diagram = st.checkbox("Show Process State Diagram", value=True)

    processes = st.session_state.processes
    if not processes:
        st.warning("Please configure processes in the sidebar.")
        return

    process_df = pd.DataFrame([
        {"Process": p.pid, "Arrival Time": p.arrival_time, 
         "Burst Time": p.burst_time, "Priority": p.priority, "Queue Level": p.queue_level}
        for p in processes
    ])
    
    st.subheader("Process Table")
    st.dataframe(process_df, use_container_width=True, hide_index=True)

    scheduler = CPUScheduler(processes)

    if "Preemptive vs Non-Preemptive" in algorithm:
        st.markdown("---")
        st.subheader("Preemptive vs Non-Preemptive Comparison")
        
        comparisons = {
            "SJF (Non-Preemptive)": CPUScheduler(processes).sjf(),
            "SRTF (Preemptive)": CPUScheduler(processes).srtf(),
            "Priority (Non-Preemptive)": CPUScheduler(processes).priority_scheduling(False),
            "Priority (Preemptive)": CPUScheduler(processes).priority_scheduling(True)
        }
        
        comparison_data = []
        for name, (procs, gantt, steps) in comparisons.items():
            metrics = calculate_metrics(procs)
            is_preemptive = "Preemptive" in name and "Non" not in name
            comparison_data.append({
                "Algorithm": name,
                "Type": "Preemptive" if is_preemptive else "Non-Preemptive",
                "Avg Waiting Time": metrics['avg_waiting_time'],
                "Avg Turnaround Time": metrics['avg_turnaround_time'],
                "Avg Response Time": metrics['avg_response_time'],
                "CPU Utilization": metrics['cpu_utilization']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        st.dataframe(
            comparison_df.style.format({
                "Avg Waiting Time": "{:.2f}",
                "Avg Turnaround Time": "{:.2f}",
                "Avg Response Time": "{:.2f}",
                "CPU Utilization": "{:.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                comparison_df, 
                x="Algorithm", 
                y="Avg Response Time",
                color="Type",
                title="Response Time: Preemptive vs Non-Preemptive",
                color_discrete_map={"Preemptive": "#667eea", "Non-Preemptive": "#764ba2"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                comparison_df, 
                x="Algorithm", 
                y="Avg Waiting Time",
                color="Type",
                title="Waiting Time: Preemptive vs Non-Preemptive",
                color_discrete_map={"Preemptive": "#667eea", "Non-Preemptive": "#764ba2"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Gantt Chart Comparison")
        
        for name, (procs, gantt, steps) in comparisons.items():
            fig = create_gantt_chart(gantt, f"{name}")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    elif "Compare All" in algorithm:
        st.markdown("---")
        st.subheader("Algorithm Comparison")
        
        algorithms = {
            "FCFS": scheduler.fcfs(),
            "SJF": CPUScheduler(processes).sjf(),
            "SRTF": CPUScheduler(processes).srtf(),
            f"Round Robin (Q={quantum})": CPUScheduler(processes).round_robin(quantum),
            "Priority (Non-Preemptive)": CPUScheduler(processes).priority_scheduling(False),
            "Priority (Preemptive)": CPUScheduler(processes).priority_scheduling(True),
            "Multilevel Queue": CPUScheduler(processes).multilevel_queue(queue_quantums)
        }
        
        comparison_data = []
        for name, (procs, gantt, steps) in algorithms.items():
            metrics = calculate_metrics(procs)
            comparison_data.append({
                "Algorithm": name,
                "Avg Waiting Time": f"{metrics['avg_waiting_time']:.2f}",
                "Avg Turnaround Time": f"{metrics['avg_turnaround_time']:.2f}",
                "Avg Response Time": f"{metrics['avg_response_time']:.2f}",
                "CPU Utilization": f"{metrics['cpu_utilization']:.1f}%",
                "Throughput": f"{metrics['throughput']:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        metrics_for_chart = []
        for name, (procs, gantt, steps) in algorithms.items():
            metrics = calculate_metrics(procs)
            metrics_for_chart.append({
                "Algorithm": name,
                "Avg Waiting Time": metrics['avg_waiting_time'],
                "Avg Turnaround Time": metrics['avg_turnaround_time']
            })
        
        chart_df = pd.DataFrame(metrics_for_chart)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_wait = px.bar(chart_df, x="Algorithm", y="Avg Waiting Time",
                             title="Average Waiting Time Comparison",
                             color="Avg Waiting Time",
                             color_continuous_scale="Viridis")
            fig_wait.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_wait, use_container_width=True)
        
        with col2:
            fig_turn = px.bar(chart_df, x="Algorithm", y="Avg Turnaround Time",
                             title="Average Turnaround Time Comparison",
                             color="Avg Turnaround Time",
                             color_continuous_scale="Plasma")
            fig_turn.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_turn, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Gantt Charts for All Algorithms")
        
        for name, (procs, gantt, steps) in algorithms.items():
            fig = create_gantt_chart(gantt, f"{name} - Gantt Chart")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            result_df = pd.DataFrame([
                {"Process": p.pid, "Completion Time": p.completion_time,
                 "Turnaround Time": p.turnaround_time, "Waiting Time": p.waiting_time,
                 "Response Time": p.response_time}
                for p in procs
            ])
            with st.expander(f"View {name} Details"):
                st.dataframe(result_df, use_container_width=True, hide_index=True)
    else:
        if "FCFS" in algorithm:
            result_processes, gantt, steps = scheduler.fcfs()
            algo_name = "First Come First Serve (FCFS)"
        elif "SJF" in algorithm:
            result_processes, gantt, steps = scheduler.sjf()
            algo_name = "Shortest Job First (SJF)"
        elif "SRTF" in algorithm:
            result_processes, gantt, steps = scheduler.srtf()
            algo_name = "Shortest Remaining Time First (SRTF)"
        elif "Round Robin" in algorithm:
            result_processes, gantt, steps = scheduler.round_robin(quantum)
            algo_name = f"Round Robin (Quantum = {quantum})"
        elif "Priority (Non-Preemptive)" in algorithm:
            result_processes, gantt, steps = scheduler.priority_scheduling(False)
            algo_name = "Priority Scheduling (Non-Preemptive)"
        elif "Priority (Preemptive)" in algorithm:
            result_processes, gantt, steps = scheduler.priority_scheduling(True)
            algo_name = "Priority Scheduling (Preemptive)"
        elif "Multilevel" in algorithm:
            result_processes, gantt, steps = scheduler.multilevel_queue(queue_quantums)
            algo_name = "Multilevel Queue Scheduling"
        else:
            result_processes, gantt, steps = scheduler.fcfs()
            algo_name = "First Come First Serve (FCFS)"

        st.markdown("---")
        st.subheader(f"Results: {algo_name}")
        
        metrics = calculate_metrics(result_processes)
        display_metrics(metrics)
        
        if show_steps and steps:
            st.markdown("---")
            display_step_by_step(steps, gantt)
        
        st.markdown("---")
        st.subheader("Gantt Chart")
        fig = create_gantt_chart(gantt, f"{algo_name} - Gantt Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        if show_state_diagram:
            st.markdown("---")
            st.subheader("Process State Diagram")
            state_fig = create_state_diagram(result_processes)
            st.plotly_chart(state_fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Process Results")
        result_df = pd.DataFrame([
            {"Process": p.pid, 
             "Arrival Time": p.arrival_time,
             "Burst Time": p.burst_time,
             "Priority": p.priority,
             "Queue Level": p.queue_level,
             "Completion Time": p.completion_time,
             "Turnaround Time": p.turnaround_time, 
             "Waiting Time": p.waiting_time,
             "Response Time": p.response_time}
            for p in result_processes
        ])
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        
        csv_data = export_to_csv(processes, result_processes)
        st.download_button(
            "Export Results to CSV",
            csv_data,
            "scheduling_results.csv",
            "text/csv"
        )
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_times = go.Figure(data=[
                go.Bar(name='Waiting Time', x=[p.pid for p in result_processes], 
                      y=[p.waiting_time for p in result_processes], marker_color='#667eea'),
                go.Bar(name='Turnaround Time', x=[p.pid for p in result_processes], 
                      y=[p.turnaround_time for p in result_processes], marker_color='#764ba2')
            ])
            fig_times.update_layout(
                title="Waiting Time vs Turnaround Time",
                barmode='group',
                xaxis_title="Process",
                yaxis_title="Time Units",
                height=400
            )
            st.plotly_chart(fig_times, use_container_width=True)
        
        with col2:
            fig_pie = go.Figure(data=[go.Pie(
                labels=[p.pid for p in result_processes],
                values=[p.burst_time for p in result_processes],
                hole=.4,
                marker_colors=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', 
                              '#00f2fe', '#43e97b', '#fa709a', '#fee140', '#30cfd0']
            )])
            fig_pie.update_layout(
                title="CPU Time Distribution",
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    with st.expander("Algorithm Information"):
        st.markdown("""
        ### CPU Scheduling Algorithms
        
        **1. FCFS (First Come First Serve)**
        - Non-preemptive algorithm
        - Processes are executed in order of arrival
        - Simple but may cause convoy effect
        
        **2. SJF (Shortest Job First)**
        - Non-preemptive algorithm
        - Process with smallest burst time executes first
        - Optimal for minimizing average waiting time
        
        **3. SRTF (Shortest Remaining Time First)**
        - Preemptive version of SJF
        - Always runs process with least remaining time
        - Better response time than SJF
        
        **4. Round Robin**
        - Preemptive algorithm with time quantum
        - Each process gets CPU for a fixed time slice
        - Good for time-sharing systems
        
        **5. Priority Scheduling**
        - Can be preemptive or non-preemptive
        - Lower priority number = higher priority
        - May cause starvation of low priority processes
        
        **6. Multilevel Queue Scheduling**
        - Multiple queues with different priority levels
        - Each queue can have its own scheduling algorithm
        - Processes are permanently assigned to a queue
        
        ### Key Metrics
        
        - **Waiting Time**: Time spent waiting in ready queue
        - **Turnaround Time**: Total time from arrival to completion
        - **Response Time**: Time from arrival to first CPU allocation
        - **CPU Utilization**: Percentage of time CPU is busy
        - **Throughput**: Number of processes completed per unit time
        """)


if __name__ == "__main__":
    main()
