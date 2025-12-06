import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Tuple
import copy

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
</style>
""", unsafe_allow_html=True)


@dataclass
class Process:
    pid: str
    arrival_time: int
    burst_time: int
    priority: int = 0
    remaining_time: int = 0
    completion_time: int = 0
    waiting_time: int = 0
    turnaround_time: int = 0
    response_time: int = -1

    def __post_init__(self):
        self.remaining_time = self.burst_time


class CPUScheduler:
    def __init__(self, processes: List[Process]):
        self.processes = [copy.deepcopy(p) for p in processes]
        self.gantt_chart = []
        self.time = 0

    def fcfs(self) -> Tuple[List[Process], List[dict]]:
        processes = sorted(self.processes, key=lambda x: (x.arrival_time, x.pid))
        gantt = []
        current_time = 0

        for p in processes:
            if current_time < p.arrival_time:
                gantt.append({"Process": "Idle", "Start": current_time, "End": p.arrival_time})
                current_time = p.arrival_time

            p.response_time = current_time - p.arrival_time
            start = current_time
            current_time += p.burst_time
            p.completion_time = current_time
            p.turnaround_time = p.completion_time - p.arrival_time
            p.waiting_time = p.turnaround_time - p.burst_time

            gantt.append({"Process": p.pid, "Start": start, "End": current_time})

        return processes, gantt

    def sjf(self) -> Tuple[List[Process], List[dict]]:
        processes = [copy.deepcopy(p) for p in self.processes]
        n = len(processes)
        completed = 0
        current_time = 0
        gantt = []
        is_completed = [False] * n

        while completed < n:
            ready = []
            for i, p in enumerate(processes):
                if p.arrival_time <= current_time and not is_completed[i]:
                    ready.append((i, p))

            if not ready:
                next_arrival = min(p.arrival_time for i, p in enumerate(processes) if not is_completed[i])
                gantt.append({"Process": "Idle", "Start": current_time, "End": next_arrival})
                current_time = next_arrival
                continue

            idx, process = min(ready, key=lambda x: (x[1].burst_time, x[1].arrival_time))
            
            process.response_time = current_time - process.arrival_time
            start = current_time
            current_time += process.burst_time
            process.completion_time = current_time
            process.turnaround_time = process.completion_time - process.arrival_time
            process.waiting_time = process.turnaround_time - process.burst_time

            gantt.append({"Process": process.pid, "Start": start, "End": current_time})
            is_completed[idx] = True
            completed += 1

        return processes, gantt

    def srtf(self) -> Tuple[List[Process], List[dict]]:
        processes = [copy.deepcopy(p) for p in self.processes]
        n = len(processes)
        current_time = 0
        completed = 0
        gantt = []
        prev_process = None
        prev_start = 0

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

            if current.response_time == -1:
                current.response_time = current_time - current.arrival_time

            current.remaining_time -= 1
            current_time += 1

            if current.remaining_time == 0:
                current.completion_time = current_time
                current.turnaround_time = current.completion_time - current.arrival_time
                current.waiting_time = current.turnaround_time - current.burst_time
                completed += 1

        if prev_process and prev_process != "Idle":
            gantt.append({"Process": prev_process, "Start": prev_start, "End": current_time})

        return processes, gantt

    def round_robin(self, quantum: int) -> Tuple[List[Process], List[dict]]:
        processes = [copy.deepcopy(p) for p in self.processes]
        n = len(processes)
        queue = []
        current_time = 0
        gantt = []
        completed = 0
        arrived = [False] * n

        sorted_indices = sorted(range(n), key=lambda i: processes[i].arrival_time)

        while completed < n:
            for i in sorted_indices:
                if not arrived[i] and processes[i].arrival_time <= current_time:
                    queue.append(i)
                    arrived[i] = True

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

            exec_time = min(quantum, p.remaining_time)
            start = current_time
            current_time += exec_time
            p.remaining_time -= exec_time

            gantt.append({"Process": p.pid, "Start": start, "End": current_time})

            for i in sorted_indices:
                if not arrived[i] and processes[i].arrival_time <= current_time:
                    queue.append(i)
                    arrived[i] = True

            if p.remaining_time > 0:
                queue.append(idx)
            else:
                p.completion_time = current_time
                p.turnaround_time = p.completion_time - p.arrival_time
                p.waiting_time = p.turnaround_time - p.burst_time
                completed += 1

        return processes, gantt

    def priority_scheduling(self, preemptive: bool = False) -> Tuple[List[Process], List[dict]]:
        processes = [copy.deepcopy(p) for p in self.processes]
        n = len(processes)
        current_time = 0
        completed = 0
        gantt = []
        is_completed = [False] * n

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

                if current.response_time == -1:
                    current.response_time = current_time - current.arrival_time

                current.remaining_time -= 1
                current_time += 1

                if current.remaining_time == 0:
                    current.completion_time = current_time
                    current.turnaround_time = current.completion_time - current.arrival_time
                    current.waiting_time = current.turnaround_time - current.burst_time
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
                
                process.response_time = current_time - process.arrival_time
                start = current_time
                current_time += process.burst_time
                process.completion_time = current_time
                process.turnaround_time = process.completion_time - process.arrival_time
                process.waiting_time = process.turnaround_time - process.burst_time

                gantt.append({"Process": process.pid, "Start": start, "End": current_time})
                is_completed[idx] = True
                completed += 1

        return processes, gantt


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


def main():
    st.markdown('<h1 class="main-header">CPU Scheduling Simulator</h1>', unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.header("Process Configuration")
        
        num_processes = st.number_input("Number of Processes", min_value=1, max_value=10, value=4)
        
        st.subheader("Process Details")
        
        processes = []
        for i in range(num_processes):
            with st.expander(f"Process P{i+1}", expanded=(i == 0)):
                col1, col2, col3 = st.columns(3)
                with col1:
                    arrival = st.number_input(f"Arrival", min_value=0, value=i, key=f"arr_{i}")
                with col2:
                    burst = st.number_input(f"Burst", min_value=1, value=max(1, 5-i), key=f"burst_{i}")
                with col3:
                    priority = st.number_input(f"Priority", min_value=1, value=i+1, key=f"prio_{i}")
                
                processes.append(Process(
                    pid=f"P{i+1}",
                    arrival_time=arrival,
                    burst_time=burst,
                    priority=priority
                ))
        
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
             "Compare All"]
        )
        
        quantum = 2
        if "Round Robin" in algorithm:
            quantum = st.number_input("Time Quantum", min_value=1, max_value=10, value=2)

    process_df = pd.DataFrame([
        {"Process": p.pid, "Arrival Time": p.arrival_time, 
         "Burst Time": p.burst_time, "Priority": p.priority}
        for p in processes
    ])
    
    st.subheader("Process Table")
    st.dataframe(process_df, use_container_width=True, hide_index=True)

    scheduler = CPUScheduler(processes)

    if "Compare All" in algorithm:
        st.markdown("---")
        st.subheader("Algorithm Comparison")
        
        algorithms = {
            "FCFS": scheduler.fcfs(),
            "SJF": CPUScheduler(processes).sjf(),
            "SRTF": CPUScheduler(processes).srtf(),
            f"Round Robin (Q={quantum})": CPUScheduler(processes).round_robin(quantum),
            "Priority (Non-Preemptive)": CPUScheduler(processes).priority_scheduling(False),
            "Priority (Preemptive)": CPUScheduler(processes).priority_scheduling(True)
        }
        
        comparison_data = []
        for name, (procs, gantt) in algorithms.items():
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
        for name, (procs, gantt) in algorithms.items():
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
        
        for name, (procs, gantt) in algorithms.items():
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
            result_processes, gantt = scheduler.fcfs()
            algo_name = "First Come First Serve (FCFS)"
        elif "SJF" in algorithm:
            result_processes, gantt = scheduler.sjf()
            algo_name = "Shortest Job First (SJF)"
        elif "SRTF" in algorithm:
            result_processes, gantt = scheduler.srtf()
            algo_name = "Shortest Remaining Time First (SRTF)"
        elif "Round Robin" in algorithm:
            result_processes, gantt = scheduler.round_robin(quantum)
            algo_name = f"Round Robin (Quantum = {quantum})"
        elif "Priority (Non-Preemptive)" in algorithm:
            result_processes, gantt = scheduler.priority_scheduling(False)
            algo_name = "Priority Scheduling (Non-Preemptive)"
        else:
            result_processes, gantt = scheduler.priority_scheduling(True)
            algo_name = "Priority Scheduling (Preemptive)"

        st.markdown("---")
        st.subheader(f"Results: {algo_name}")
        
        metrics = calculate_metrics(result_processes)
        display_metrics(metrics)
        
        st.markdown("---")
        st.subheader("Gantt Chart")
        fig = create_gantt_chart(gantt, f"{algo_name} - Gantt Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Process Results")
        result_df = pd.DataFrame([
            {"Process": p.pid, 
             "Arrival Time": p.arrival_time,
             "Burst Time": p.burst_time,
             "Priority": p.priority,
             "Completion Time": p.completion_time,
             "Turnaround Time": p.turnaround_time, 
             "Waiting Time": p.waiting_time,
             "Response Time": p.response_time}
            for p in result_processes
        ])
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        
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
        """)


if __name__ == "__main__":
    main()
