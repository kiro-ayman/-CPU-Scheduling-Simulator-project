# CPU Scheduling Simulator

## Overview

This is a CPU scheduling simulator built with Streamlit, a Python framework for creating interactive web applications. The application visualizes and analyzes different CPU scheduling algorithms, allowing users to compare performance metrics across various scheduling strategies. The simulator uses Plotly for interactive data visualization and Pandas for data manipulation.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit
- **Rationale**: Streamlit enables rapid development of data-centric web applications with minimal frontend code. It provides automatic UI rendering from Python code, making it ideal for simulation and visualization tools.
- **Pros**: Fast prototyping, Python-native, built-in state management
- **Cons**: Limited customization compared to traditional frontend frameworks

### UI/UX Design
- **Custom Styling**: CSS injection through `st.markdown()` for enhanced visual presentation
- **Design Pattern**: Gradient-based color scheme (purple gradient: #667eea to #764ba2) for modern aesthetic
- **Layout**: Wide layout configuration with expanded sidebar for controls and inputs
- **Component Organization**: Tab-based interface for organizing different views and scheduling algorithms

### Data Processing Layer
- **Data Structures**: Python dataclasses for structured process representation
- **Rationale**: Dataclasses provide type hints and automatic initialization, making process data more maintainable and self-documenting
- **Data Manipulation**: Pandas DataFrames for tabular process data and metrics calculation
- **Pros**: Type safety, clean data structure definition, efficient data operations

### Visualization Layer
- **Library**: Plotly (both Express and Graph Objects)
- **Rationale**: Plotly provides interactive, publication-quality visualizations suitable for Gantt charts and performance metrics
- **Alternatives Considered**: Matplotlib (less interactive), Altair (limited chart types)
- **Pros**: Interactive charts, web-ready output, extensive chart types
- **Use Cases**: 
  - Gantt charts for process scheduling timelines
  - Performance metric comparisons across algorithms

### Application Structure
- **Entry Point**: `main.py` serves as a minimal entry point
- **Core Application**: `app.py` contains the primary scheduling simulator logic
- **Separation Rationale**: Allows for modular execution and potential CLI tool development separate from the web interface

### Scheduling Algorithms (Anticipated)
Based on the application type, the system likely implements multiple CPU scheduling algorithms:
- First Come First Serve (FCFS)
- Shortest Job First (SJF)
- Round Robin
- Priority Scheduling
- Shortest Remaining Time First (SRTF)

Each algorithm would calculate metrics such as:
- Average waiting time
- Average turnaround time
- CPU utilization
- Throughput

## External Dependencies

### Core Frameworks
- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive data visualization (both `plotly.express` and `plotly.graph_objects`)

### Python Standard Library
- **dataclasses**: Structured data representation for processes
- **typing**: Type hints for better code documentation and IDE support
- **copy**: Deep copying of process objects to preserve original state across simulations

### Development Environment
- **Replit**: Cloud-based development and hosting platform
- **Nix**: Package management (indicated by repl-nix-workspace reference)

### No External Services
The application appears to be self-contained with no external API integrations, databases, or third-party authentication services. All data is processed in-memory during user sessions.