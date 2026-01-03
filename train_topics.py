from src.topics_bertopic import train_bertopic_from_chroma

if __name__ == "__main__":
    train_bertopic_from_chroma(
        model_dir="output/bertopic_model.pkl",
        embedding_model_name="all-MiniLM-L6-v2",
        max_docs=None,          # puedes poner 5000 si tarda mucho
        min_topic_size=30,      # ajustable
    )
    print("✅ BERTopic trained and saved to output/bertopic_model.pkl")

    