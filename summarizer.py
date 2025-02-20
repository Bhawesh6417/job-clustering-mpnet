import google.generativeai as genai
import pandas as pd

# Configure API key
genai.configure(api_key="AIzaSyB8ElKvrPTYaV9kG2q4AsTXYCc6fTIk0Uo")

def analyze_cluster_differences(data):
    """Asks Gemini to explain why different clusters were formed based on job descriptions."""
    prompt = (
        "I have extracted job-related data and applied clustering on it. "
        "This resulted in multiple clusters and sub-clusters. I want to understand why these clusters were formed. "
        "Below is the extracted data containing job descriptions, clusters, and sub-clusters.\n\n"
        "Analyze the job descriptions and explain the key differences between clusters. "
        "What patterns, themes, or similarities exist within each cluster? "
        "How do the sub-clusters refine the grouping further?\n\n"
        f"Data:\n{data}"
    )

    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content(prompt)
    return response.text if response else "Analysis failed."

def process_file(file_path):
    """Processes the CSV file and extracts job descriptions along with cluster info."""
    df = pd.read_csv(file_path)

    # Ensure required columns exist
    required_columns = {"job_description", "cluster", "sub_cluster"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file must contain {required_columns} columns.")

    # Combine job descriptions with clusters for analysis
    cluster_data = df[["job_description", "cluster", "sub_cluster"]].to_string(index=False)

    # Get Gemini's explanation of cluster formation
    cluster_analysis = analyze_cluster_differences(cluster_data)

    return cluster_analysis

# File path to MPNet's extracted CSV output
file_path = "MPNet_output_file_extracted.csv"
cluster_analysis = process_file(file_path)

# Save the analysis to a file
with open("cluster_analysis.txt", "w", encoding="utf-8") as output_file:
    output_file.write(cluster_analysis)

print("Cluster analysis complete! Check cluster_analysis.txt.")
