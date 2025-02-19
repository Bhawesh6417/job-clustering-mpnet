import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model (MPNet)
model = SentenceTransformer("all-mpnet-base-v2")

# Path to job JSON files
job_folder = "C:\\Users\\Bhawesh\\Downloads\\drive-download-20250216T061214Z-001\\jobs"

# Load job postings from JSON files
def load_jobs(folder):
    jobs = []
    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    jobs.extend(item.get("jobs", []))  # Handle missing 'jobs' key gracefully
    return pd.DataFrame(jobs)

df = load_jobs(job_folder)
df = df[["title", "skillset", "job_description"]]
df["skillset"] = df["skillset"].apply(lambda x: x.split(", ") if isinstance(x, str) else [])

# Count skill occurrences
all_skills = [skill for skills in df["skillset"] for skill in skills]
skill_counts = Counter(all_skills)
skill_df = pd.DataFrame(skill_counts.items(), columns=["Skill", "Count"]).sort_values(by="Count", ascending=False)
skill_df = skill_df[skill_df["Skill"] != ""]

# Plot top 20 skills
plt.figure(figsize=(12, 6))
plt.barh(skill_df["Skill"][:20], skill_df["Count"][:20], color="skyblue")
plt.xlabel("Number of Jobs Requiring Skill")
plt.ylabel("Skill")
plt.title("Top 20 Most In-Demand Skills")
plt.gca().invert_yaxis()
plt.show()

# Function to get embeddings using MPNet
def get_mpnet_embeddings(text):
    try:
        if not text.strip():
            return np.zeros(768)  # MPNet embedding size is 768
        return model.encode(text, convert_to_numpy=True)
    except Exception as e:
        print(f"Error in embedding generation: {e}")
        return np.zeros(768)  # Fallback in case of error

# Load or generate embeddings
embedding_file = "job_embeddings_mpnet.npy"
if os.path.exists(embedding_file):
    X = np.load(embedding_file)
else:
    X = np.array([get_mpnet_embeddings(desc) if isinstance(desc, str) and desc.strip() else np.zeros(768) 
                  for desc in tqdm(df["job_description"].fillna(""), desc="Generating Embeddings")])
    np.save(embedding_file, X)  # Save embeddings for reuse

# Scale embeddings
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal clusters using Elbow Method
inertia = []
range_of_clusters = range(1, 11)
for n_clusters in range_of_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # Correct n_init
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(range_of_clusters, inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Perform clustering
target_clusters = 5  # Adjust based on elbow graph
kmeans = KMeans(n_clusters=target_clusters, random_state=42, n_init=10)  # Correct n_init
df["cluster"] = kmeans.fit_predict(X_scaled)

# Save clustering results to a file
output_file = "MPNet's_output_file.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Job Clustering Results\n\n")
    
    for i in range(target_clusters):
        df_sub = df[df["cluster"] == i].copy()
        f.write(f"Main Cluster {i} ({len(df_sub)} jobs):\n")
        
        for _, row in df_sub.iterrows():
            f.write(f"  - Title: {row['title']}\n")
            f.write(f"    Skills: {', '.join(row['skillset'])}\n")
            f.write(f"    Description: {row['job_description'][:200]}...\n\n")  # Limiting long descriptions

        f.write("=" * 50 + "\n\n")

print(f"Clustering results saved to: {output_file}")

# Process clusters
for i in range(target_clusters):
    df_sub = df[df["cluster"] == i].copy()
    print(f"\nMain Cluster {i} ({len(df_sub)} jobs):")
    print(df_sub[["title", "skillset"]].head(5))
    
    if df_sub.shape[0] < 3:
        continue

    df_sub["skill_text"] = df_sub["skillset"].apply(lambda x: " ".join(x))
    X_skills = np.array([get_mpnet_embeddings(skill_text) for skill_text in df_sub["skill_text"]])
    X_skills_scaled = scaler.fit_transform(X_skills)
    
    num_sub_clusters = min(3, df_sub.shape[0])
    kmeans_skills = KMeans(n_clusters=num_sub_clusters, random_state=42, n_init=10)  # Correct n_init
    df_sub["sub_cluster"] = kmeans_skills.fit_predict(X_skills_scaled)
    
    print("\n   Job Count in Each Sub-Cluster:")
    print(df_sub["sub_cluster"].value_counts())
    
    with open(output_file, "a", encoding="utf-8") as f:
        for sc in range(num_sub_clusters):
            df_sub_sc = df_sub[df_sub["sub_cluster"] == sc]
            f.write(f"\n   Sub-Cluster {sc} within Cluster {i} ({len(df_sub_sc)} jobs):\n")
            for _, row in df_sub_sc.iterrows():
                f.write(f"    - Title: {row['title']}\n")
                f.write(f"      Skills: {', '.join(row['skillset'])}\n")
                f.write(f"      Description: {row['job_description'][:200]}...\n\n")

# Word Cloud for Job Descriptions
text = " ".join(df["job_description"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Job Descriptions")
plt.show()
