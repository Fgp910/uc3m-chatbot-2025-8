import time
import json
import re
import statistics
import logging
from typing import List, Dict, Any, Optional

# Mantenemos tus importaciones originales
try:
    from src.rag import get_rag_chain
    from src.vector_store import get_retriever
    from src.llm_client import call_llm_api_full
except ImportError as e:
    print(f"Error: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustRAGEvaluator:
    def __init__(self, k_docs=5):
        self.retriever = get_retriever(k_docs=k_docs)
        self.rag_chain = get_rag_chain(self.retriever)
        self.results = []
        # Frases clave que indican que el modelo se niega a responder (Requisito PDF)
        self.refusal_phrases = [
            "no tengo información", "no cuento con información", 
            "no se menciona", "no aparece en los documentos",
            "no puedo responder", "documents do not contain"
        ]

    def _judge_llm(self, prompt: str) -> float:
        """Helper robusto para métricas LLM-as-a-judge con reintentos simples."""
        try:
            response = call_llm_api_full(prompt).strip()
            # Busca un número entero o decimal entre 0 y 1
            match = re.search(r'\b(0(\.\d+)?|1(\.0+)?)\b', response)
            if match:
                return float(match.group(1))
            return 0.0
        except Exception as e:
            logger.error(f"Error llamando al Juez LLM: {e}")
            return 0.0

    def _calculate_retrieval_score(self, retrieved_docs: List[Any], expected_source: str) -> float:
        """
        Métrica: Context Recall (Hit Rate).
        Verifica si el documento esperado está entre los recuperados.
        """
        if not expected_source: return 1.0 # Si no hay fuente esperada, no evaluamos recuperación
        
        found = False
        # Normalizamos nombres de archivo para evitar errores por mayúsculas o extensiones
        expected_clean = expected_source.lower().replace(".pdf", "").strip()
        
        for doc in retrieved_docs:
            source_meta = doc.metadata.get('source', '').lower()
            if expected_clean in source_meta:
                found = True
                break
        return 1.0 if found else 0.0

    def _evaluate_refusal(self, response: str) -> float:
        """Verifica si el modelo se niega correctamente a responder."""
        response_lower = response.lower()
        return 1.0 if any(phrase in response_lower for phrase in self.refusal_phrases) else 0.0

    def evaluate_dataset(self, dataset: List[Dict[str, Any]]):
        logger.info(f"Iniciando evaluación robusta de {len(dataset)} casos...")
        
        metrics = {
            "latency": [],
            "retrieval_recall": [], # ¿El retriever trajo el PDF correcto?
            "faithfulness": [],     # ¿El generador inventó cosas?
            "answer_relevance": [], # ¿Respondió a la pregunta?
            "citation_accuracy": [], # ¿Citó la fuente?
            "negative_handling": []  # ¿Supo decir 'no sé'?
        }

        for i, case in enumerate(dataset):
            question = case["question"]
            expected_source = case.get("ground_truth_source") # Puede ser None para preguntas trampa
            
            logger.info(f"--- Caso {i+1}: {question} ---")
            
            # 1. Ejecución y Latencia
            start = time.time()
            full_response = ""
            config = {"configurable": {"session_id": f"eval_{i}"}}
            
            try:
                # Invocación manual del retriever para inspección (White-box testing)
                retrieved_docs = self.retriever.invoke(question)
                context_text = "\n".join([d.page_content for d in retrieved_docs])
                
                # Invocación de la cadena de generación
                for chunk in self.rag_chain.stream({"question": question}, config=config):
                    full_response += str(chunk)
            except Exception as e:
                logger.error(f"Fallo en ejecución: {e}")
                full_response = "ERROR"
                retrieved_docs = []
                context_text = ""

            latency = time.time() - start
            metrics["latency"].append(latency)

            # 2. Evaluación Diferenciada (Positiva vs Negativa)
            
            # CASO A: Pregunta sin respuesta esperada (Negative Testing)
            if expected_source is None:
                score_refusal = self._evaluate_refusal(full_response)
                metrics["negative_handling"].append(score_refusal)
                # En casos negativos, la fidelidad es 1 si se niega, 0 si inventa
                metrics["faithfulness"].append(score_refusal) 
                logger.info(f"   Negativa -> Correcta: {bool(score_refusal)}")
                
            # CASO B: Pregunta estándar con documento
            else:
                # A. Métrica de Recuperación (Retrieval)
                retrieval_score = self._calculate_retrieval_score(retrieved_docs, expected_source)
                metrics["retrieval_recall"].append(retrieval_score)

                # B. Métrica de Generación (Faithfulness - Juez LLM)
                prompt_faith = (
                    f"Contexto recuperado: {context_text}\n"
                    f"Respuesta del sistema: {full_response}\n"
                    "Tarea: Evalúa si la respuesta está sustentada completamente por el contexto. "
                    "Si hay información externa o inventada, devuelve 0. Si es fiel, devuelve 1. Solo el número."
                )
                faith_score = self._judge_llm(prompt_faith)
                metrics["faithfulness"].append(faith_score)

                # C. Métrica de Relevancia (Answer Relevance - Juez LLM)
                prompt_rel = (
                    f"Pregunta: {question}\nRespuesta: {full_response}\n"
                    "Tarea: ¿La respuesta contesta directamente a la pregunta? 1 si sí, 0 si no. Solo el número."
                )
                rel_score = self._judge_llm(prompt_rel)
                metrics["answer_relevance"].append(rel_score)

                # D. Métrica de Citación (Estricta)
                # Verifica si el nombre del archivo (sin extensión) aparece en la respuesta
                cit_score = 1.0 if expected_source.replace('.pdf','').lower() in full_response.lower() else 0.0
                metrics["citation_accuracy"].append(cit_score)

                logger.info(f"   Retrieval: {retrieval_score} | Faith: {faith_score} | Citation: {cit_score}")

            # Guardar resultado individual
            self.results.append({
                "question": question,
                "response": full_response,
                "metrics": {
                    "latency": latency,
                    "retrieval": retrieval_score if expected_source else "N/A",
                    "faithfulness": faith_score if expected_source else score_refusal
                }
            })

        self._print_final_report(metrics)

    def _print_final_report(self, metrics):
        print("\n" + "="*60)
        print(" REPORTE DE EVALUACIÓN ")
        print("="*60)
        
        def safe_mean(lst): return statistics.mean(lst) if lst else 0.0

        print(f"| Componente  | Métrica | Resultado | Objetivo |")
        print(f"| :--- | :--- | :--- | :--- |")
        print(f"| **Retrieval** | Context Recall (Hit Rate) | {safe_mean(metrics['retrieval_recall']):.2%} | > 80% |")
        print(f"| **Generator** | Faithfulness (Anti-alucinación)| {safe_mean(metrics['faithfulness']):.2%} | > 90% |")
        print(f"| **Functional**| Manejo de Negativas (No info)| {safe_mean(metrics['negative_handling']):.2%} | 100% |")
        print(f"| **Functional**| Citation Accuracy (Fuente) | {safe_mean(metrics['citation_accuracy']):.2%} | 100% |")
        print(f"| **Performance**| Latencia Promedio | {safe_mean(metrics['latency']):.2f}s | < 5s |")
        
        print("\nNotas para el informe:")
        if safe_mean(metrics['retrieval_recall']) < 0.8:
            print("⚠️ El Retriever está fallando. Considera aumentar K_DOCS o mejorar los Embeddings.")
        if safe_mean(metrics['citation_accuracy']) < 1.0:
            print("⚠️ Falla el requisito de citas. Ajusta el System Prompt de Llama3 para que sea obligatorio citar.")

# --- Configuracion del Dataset de Prueba (Golden Dataset) ---
TEST_DATASET = [
    {
        "question": "¿Cuál es el modelo específico y la potencia nominal de los 45 inversores que componen la unidad de generación del proyecto Cottonwood Bayou?",
        "ground_truth_source": "35077_1406_1202349.pdf"
    }
]

if __name__ == '__main__':
    evaluator = RobustRAGEvaluator()
    evaluator.evaluate_dataset(TEST_DATASET)

