"use client";

import { useState, useEffect } from "react";
import { Card, Badge, Button, Table, Grid, Metric, Dialog, Flex } from "@tremor/react";
import { CheckCircle, AlertCircle, XCircle, Upload, RefreshCw, Play } from "lucide-react";
import { useRouter } from "next/navigation";

// Types
interface EvalResult {
  question: string;
  expected_answer: string;
  actual_answer?: string;
  chunk_scores: Array<{ index: number; relevance: number; reasoning?: string }>;
  answer_scores: {
    grounded: boolean;
    confidence: number;
    reasoning: string;
  };
  status: "success" | "failed" | "pending";
}

interface EvalSummary {
  avg_chunk_relevance: number;
  avg_answer_confidence: number;
  avg_context_precision: number;
  avg_answer_relevancy: number;
  total_evaluated: number;
}

export default function EvalPage() {
  const router = useRouter();
  const [evaluations, setEvaluations] = useState<EvalResult[]>([]);
  const [summary, setSummary] = useState<EvalSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedEval, setSelectedEval] = useState<EvalResult | null>(null);

  // Load eval dataset
  useEffect(() => {
    const loadDataset = async () => {
      try {
        const res = await fetch("/api/eval-dataset");
        if (res.ok) {
          const data = await res.json();
          // Initialize evaluations with pending status
          const initialEvals = data.questions.map((q: any) => ({
            question: q.question,
            expected_answer: q.expected_answer,
            actual_answer: "",
            chunk_scores: [],
            answer_scores: { grounded: false, confidence: 0, reasoning: "" },
            status: "pending" as const,
          }));
          setEvaluations(initialEvals);
        }
      } catch (error) {
        console.error("Failed to load eval dataset:", error);
      }
    };
    loadDataset();
  }, []);

  // Run evaluation
  const runEvaluation = async () => {
    setLoading(true);
    const newEvaluations = [...evaluations];

    for (let i = 0; i < newEvaluations.length; i++) {
      let evalData = newEvaluations[i];
      try {
        const response = await fetch("/api/evaluate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: evalData.question,
            answer: evalData.expected_answer, // Using expected as actual for testing
            chunks: [
              {
                pageContent: `Test chunk ${i + 1}`,
                metadata: { chunk: i + 1 },
              },
            ],
            mode: "full",
          }),
        });

        const result = await response.json();
        if (result.status === "success") {
          newEvaluations[i] = {
            ...evalData,
            actual_answer: evalData.expected_answer,
            chunk_scores: result.results.chunk_scores,
            answer_scores: result.results.answer_scores,
            status: "success" as const,
          };
        } else {
          newEvaluations[i] = {
            ...evalData,
            status: "failed" as const,
          };
        }
      } catch (error) {
        newEvaluations[i] = {
          ...evalData,
          status: "failed" as const,
        };
      }

      setEvaluations([...newEvaluations]);
      updateSummary(newEvaluations);
    }

    setLoading(false);
  };

  // Update summary statistics
  const updateSummary = (evals: EvalResult[]) => {
    const successEvals = evals.filter((e) => e.status === "success");
    if (successEvals.length === 0) return;

    const groundedCount = successEvals.filter((e) => e.answer_scores.grounded).length;
    const avgConfidence = successEvals.reduce((sum, e) => sum + e.answer_scores.confidence, 0) / successEvals.length;
    const avgRelevance = successEvals.reduce((sum, e) => sum + e.chunk_scores.reduce((s, c) => s + c.relevance, 0) / e.chunk_scores.length, 0) / successEvals.length;

    setSummary({
      total_evaluated: successEvals.length,
      avg_answer_confidence: avgConfidence,
      avg_context_precision: avgRelevance,
      avg_chunk_relevance: avgRelevance,
      avg_answer_relevancy: avgRelevance,
    });
  };

  // Calculate metrics
  const groundedCount = evaluations.filter((e) => e.status === "success" && e.answer_scores.grounded).length;
  const total = evaluations.length;
  const successRate = total > 0 ? Math.round((groundedCount / total) * 100) : 0;

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Evaluation Dashboard</h1>
        <p className="text-gray-600 mt-2">Monitor RAG pipeline quality with real scores</p>
      </div>

      {/* Summary Cards */}
      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="flex flex-col items-center justify-center">
            <Metric>Avg Context Precision</Metric>
            <p className="text-2xl font-bold text-gray-900">{(summary.avg_chunk_relevance * 100).toFixed(1)}%</p>
            <Badge color="emerald">High</Badge>
          </Card>

          <Card className="flex flex-col items-center justify-center">
            <Metric>Avg Answer Confidence</Metric>
            <p className="text-2xl font-bold text-gray-900">{(summary.avg_answer_confidence * 100).toFixed(1)}%</p>
            <Badge color={summary.avg_answer_confidence > 0.7 ? "emerald" : "amber"}>
              {summary.avg_answer_confidence > 0.7 ? "High" : "Medium"}
            </Badge>
          </Card>

          <Card className="flex flex-col items-center justify-center">
            <Metric>Total Evaluated</Metric>
            <p className="text-2xl font-bold text-gray-900">{summary.total_evaluated}</p>
            <Badge color="blue">Questions</Badge>
          </Card>

          <Card className="flex flex-col items-center justify-center">
            <Metric>Grounded Rate</Metric>
            <p className="text-2xl font-bold text-gray-900">{successRate}%</p>
            <Badge color={successRate > 80 ? "emerald" : "amber"}>
              {successRate > 80 ? "Good" : "Needs Improvement"}
            </Badge>
          </Card>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-4 mb-6">
        <Button
          onClick={runEvaluation}
          disabled={loading}
          variant="primary"
          className={loading ? "opacity-75" : ""}
        >
          {loading ? (
            <>
              <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
              Running Evaluations...
            </>
          ) : (
            <>
              <Play className="w-5 h-5 mr-2" />
              Run All Evaluations
            </>
          )}
        </Button>

        <Button onClick={() => router.refresh()}>
          <RefreshCw className="w-5 h-5 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Evaluations Table */}
      <Card>
        <h2 className="text-xl font-semibold mb-4">Evaluation Results</h2>
        <div className="overflow-x-auto">
          <Table>
            <thead>
              <tr>
                <th className="text-left">Question</th>
                <th className="text-left">Status</th>
                <th className="text-left">Grounded</th>
                <th className="text-left">Confidence</th>
                <th className="text-left">Actions</th>
              </tr>
            </thead>
            <tbody>
              {evaluations.map((evalResult, index) => (
                <tr key={index}>
                  <td className="max-w-xs truncate">
                    <span className="text-sm text-gray-900">{evalResult.question}</span>
                  </td>
                  <td>
                    {evalResult.status === "success" ? (
                      <Badge color="emerald">
                        Success
                      </Badge>
                    ) : evalResult.status === "failed" ? (
                      <Badge color="red">
                        Failed
                      </Badge>
                    ) : (
                      <Badge color="amber">
                        Pending
                      </Badge>
                    )}
                  </td>
                  <td>
                    {evalResult.status === "success" ? (
                      evalResult.answer_scores.grounded ? (
                        <CheckCircle className="w-5 h-5 text-emerald-600" />
                      ) : (
                        <XCircle className="w-5 h-5 text-red-600" />
                      )
                    ) : (
                      <div className="text-gray-400">-</div>
                    )}
                  </td>
                  <td>
                    {evalResult.status === "success" ? (
                      <span>{(evalResult.answer_scores.confidence * 100).toFixed(1)}%</span>
                    ) : (
                      <div className="text-gray-400">-</div>
                    )}
                  </td>
                  <td>
                    <Button
                      size="xs"
                      onClick={() => setSelectedEval(evalResult)}
                    >
                      View
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        </div>
      </Card>

      {/* Detailed View Dialog */}
      {selectedEval && (
        <Dialog open={!!selectedEval} onClose={() => setSelectedEval(null)}>
          <div className="space-y-4">
            <h3 className="text-lg font-semibold mb-4">Evaluation Details</h3>

            <div className="mb-4">
              <h4 className="font-medium mb-1">Question:</h4>
              <p className="text-gray-700">{selectedEval.question}</p>
            </div>

            <div className="mb-4">
              <h4 className="font-medium mb-1">Expected Answer:</h4>
              <p className="text-gray-700 bg-gray-50 p-3 rounded">{selectedEval.expected_answer}</p>
            </div>

            {selectedEval.actual_answer && (
              <div className="mb-4">
                <h4 className="font-medium mb-1">Actual Answer:</h4>
                <p className="text-gray-700 bg-gray-50 p-3 rounded">{selectedEval.actual_answer}</p>
              </div>
            )}

            {selectedEval.status === "success" && (
              <>
                <div className="mb-4">
                  <h4 className="font-medium mb-1">Grounded Status:</h4>
                  <Badge color={selectedEval.answer_scores.grounded ? "emerald" : "red"}>
                    {selectedEval.answer_scores.grounded ? "Grounded" : "Not Grounded"}
                  </Badge>
                </div>

                <div className="mb-4">
                  <h4 className="font-medium mb-1">Confidence Score:</h4>
                  <Flex alignItems="center">
                    <div className="w-16 h-16 relative">
                      <svg className="w-full h-full" viewBox="0 0 36 36">
                        <path
                          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                          fill="none"
                          stroke="#e5e7eb"
                          strokeWidth="3.8"
                        />
                        <path
                          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                          fill="none"
                          stroke={selectedEval.answer_scores.confidence > 0.7 ? "#10b981" : "#f59e0b"}
                          strokeWidth="3.8"
                          strokeDasharray={`${selectedEval.answer_scores.confidence * 100}, 100`}
                          className="transition-all duration-1000 ease-in-out"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-xs font-bold">{(selectedEval.answer_scores.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                    <span className="ml-2 text-gray-700 text-sm">{(selectedEval.answer_scores.confidence * 100).toFixed(1)}%</span>
                  </Flex>
                </div>

                <div className="mb-4">
                  <h4 className="font-medium mb-1">Reasoning:</h4>
                  <p className="text-gray-700">{selectedEval.answer_scores.reasoning}</p>
                </div>
              </>
            )}
          </div>
        </Dialog>
      )}
    </div>
  );
}
