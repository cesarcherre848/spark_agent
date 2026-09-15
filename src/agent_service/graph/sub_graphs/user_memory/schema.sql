-- Schema para Memoria Semántica a Largo Plazo de Usuario (user_memory)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS user_memory (
    id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES res_users(id) ON DELETE CASCADE,
    session_id VARCHAR(64),
    content TEXT NOT NULL,
    embedding_1024 vector(1024),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índice BTree para filtrado rápido por usuario comercial
CREATE INDEX IF NOT EXISTS idx_user_memory_user_id ON user_memory(user_id);

-- Índice HNSW para búsqueda semántica acelerada por similitud de coseno
CREATE INDEX IF NOT EXISTS idx_user_memory_vector ON user_memory 
USING hnsw (embedding_1024 vector_cosine_ops);
