# Job Clustering and Skill Analysis using MPNet

This project analyzes job postings by extracting skills, performing clustering using MPNet sentence embeddings, and visualizing insights. The model groups similar jobs based on descriptions and skill sets, helping to understand trends in the job market.

## Features

- Extracts job titles, skill sets, and descriptions from JSON files.
- Uses Sentence Transformers (MPNet) to generate job embeddings.
- Performs K-Means clustering to group similar job postings.
- Determines optimal clusters using the Elbow Method.
- Generates skill frequency analysis and word clouds.
- Saves results to a text file with clustered job details.
- Summarizes cluster formation reasons based on job descriptions using Gemini AI.

## Summarizer Description

The summarizer analyzes the clustered job descriptions to explain why different clusters are formed. Using Gemini AI, it extracts key themes, skills, and role similarities to provide insights into what differentiates each cluster and sub-cluster. This helps in understanding job market segmentation and trends effectively.

## Requirements

- Python 3.7+
- Required Libraries:
  ```bash
  pip install numpy pandas matplotlib tqdm scikit-learn wordcloud sentence-transformers flask
  ```

## Usage

1. **Download Job Data**: Place JSON job files in the `jobs` folder.
2. **Run the Script**:
   ```bash
   python job_clustering.py
   ```
3. **Output Files**:
   - `MPNet_output_file.txt`: Contains job clusters and descriptions.
   - `job_embeddings_mpnet.npy`: Stores precomputed embeddings for efficiency.
   - `cluster_summary.txt`: Explanation of cluster formation reasons.

## Results

- **Top Skills Analysis**: Visualizes the most in-demand skills.
- **Clustering Report**: Groups jobs into categories based on descriptions and skills.
- **Word Cloud**: Shows frequently used words in job descriptions.
- **Cluster Summarization**: Provides reasoning behind cluster formations.

## Repository Structure

```
📂 job-clustering-mpnet
├── 📂 jobs                  # Folder containing JSON job data
├── job_clustering.py         # Main script for analysis
├── MPNet_output_file.txt     # Clustering results
├── job_embeddings_mpnet.npy  # Cached embeddings
├── summarizer.py             # Summarizes cluster formation
├── cluster_summary.txt       # Cluster formation insights
├── README.md                 # Project documentation
```

## Future Improvements

- Implement interactive visualization tools.
- Use different transformer models for embeddings.
- Integrate with a database for real-time analysis.
- Enhance summarization using LLMs for better insights.

## Output

![skills_plot](https://github.com/user-attachments/assets/e62d5c18-1c52-4b84-ba31-b1679a1fce7f)

![elbow_method_plot](https://github.com/user-attachments/assets/24df9822-ee3a-4dec-b038-9e6140880113)

![wordcloud](https://github.com/user-attachments/assets/51ef4dc5-1829-458f-8d0a-392e9b15538d)



