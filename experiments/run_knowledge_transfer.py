import argparse
import json
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.teacher import TeacherEnsemble
from src.improved_distiller import ImprovedDistiller, AdaptiveStudentNetwork
from src.distiller import AnomalyDistiller
from src.data import load_dataset, MAIN_DATASETS
from src.metrics import compute_metrics, score_correlation
from src.anomaly_taxonomy import AnomalyTaxonomist, create_taxonomy_report
from src.knowledge_decomposition import KnowledgeDecomposer, create_knowledge_report
from src.failure_analysis import FailureModeAnalyzer, create_failure_analysis_report, BoundaryStabilityAnalyzer
from src.diffusion_diagnostics import DiffusionProbe, create_diffusion_diagnostic_report
from src.fixed_baselines import run_fixed_baseline_comparison
from src.statistics import wilcoxon_signed_rank_test, friedman_test


class KnowledgeTransferExperiment:

    MIN_DATASET_SIZE = 400
    
    def __init__(self, output_dir: str = 'experiments/knowledge_transfer_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '4.0-knowledge-transfer',
                'description': 'Knowledge transfer analysis for anomaly detection distillation'
            },
            'performance': [],           
            'taxonomy_analysis': [],      
            'knowledge_decomposition': [], 
            'failure_analysis': [],       
            'diffusion_diagnostics': [], 
            'baselines': [],              
            'summary': {}                
        }

    def _save_results(self):
        
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(self._make_serializable(self.results), f, indent=2, default=str)

    def _make_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        return obj

    def run_full_analysis(self, datasets: list[str] = None, n_runs: int = 3):

        datasets = datasets or MAIN_DATASETS
        
        print("=" * 70)
        print("KNOWLEDGE TRANSFER ANALYSIS")
        print("What survives distillation? What is lost?")
        print("=" * 70)
        
        all_taxonomy_summaries = []
        all_knowledge_summaries = []
        all_failure_summaries = []
        
        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*60}")
            
            try:
                X_train, X_test, y_train, y_test = load_dataset(dataset_name)
                X_train_normal = X_train[y_train == 0]
                
                if len(X_train_normal) < self.MIN_DATASET_SIZE:
                    print(f"  Skipping: only {len(X_train_normal)} samples")
                    continue
                
                print(f"  Samples: {len(X_train)} train, {len(X_test)} test")
                print(f"  Features: {X_train.shape[1]}, Contamination: {y_test.mean():.2%}")
                
            except Exception as e:
                print(f"  Failed to load: {e}")
                continue
            

            for run_id in range(n_runs):
                print(f"\n  Run {run_id + 1}/{n_runs}")
                seed = 42 + run_id
                np.random.seed(seed)
                
                try:
                    result = self._analyze_dataset(
                        X_train, X_test, y_train, y_test, 
                        dataset_name, seed
                    )
                    
 
                    self.results['performance'].append(result['performance'])
                    self.results['taxonomy_analysis'].append(result['taxonomy'])
                    self.results['knowledge_decomposition'].append(result['knowledge'])
                    self.results['failure_analysis'].append(result['failures'])
                    self.results['diffusion_diagnostics'].append(result['diffusion'])
                    

                    all_taxonomy_summaries.append(result['taxonomy'])
                    all_knowledge_summaries.append(result['knowledge'])
                    all_failure_summaries.append(result['failures'])
                    
                    print(f"    Teacher AUC: {result['performance']['teacher_auc']:.4f}")
                    print(f"    Student AUC: {result['performance']['student_auc']:.4f}")
                    print(f"    Retention: {result['performance']['retention']:.1f}%")
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            self._save_results()
        

        self._generate_summary(all_taxonomy_summaries, all_knowledge_summaries, all_failure_summaries)
        self._save_results()
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 70)

    def _analyze_dataset(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        dataset_name: str,
        seed: int
    ) -> dict:

        X_train_normal = X_train[y_train == 0]
        

        print("    Training teacher ensemble...")
        teacher = TeacherEnsemble()
        teacher.fit(X_train_normal, epochs=50, batch_size=256)
        teacher_scores = teacher.predict_scores(X_test)
        teacher_metrics = compute_metrics(y_test, teacher_scores)
        

        print("    Distilling to student...")
        student = AdaptiveStudentNetwork(
            input_dim=X_train.shape[1],
            n_samples=len(X_train_normal),
            use_reconstruction_head=len(X_train_normal) > 1000
        )
        distiller = ImprovedDistiller(teacher, student, {'alpha': 0.1})
        distiller.distill(X_train_normal, epochs=150, patience=25)
        
        student_scores = distiller.predict(X_test)
        student_metrics = compute_metrics(y_test, student_scores)
        

        performance = {
            'dataset': dataset_name,
            'seed': seed,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X_train.shape[1],
            'contamination': float(y_test.mean()),
            'teacher_auc': teacher_metrics['auc_roc'],
            'student_auc': student_metrics['auc_roc'],
            'teacher_f1': teacher_metrics['f1'],
            'student_f1': student_metrics['f1'],
            'retention': student_metrics['auc_roc'] / teacher_metrics['auc_roc'] * 100,
            'correlation': score_correlation(teacher_scores, student_scores)
        }
        

        print("    Analyzing anomaly types...")
        taxonomy_report = create_taxonomy_report(
            X_train_normal, X_test, y_test,
            teacher_scores, student_scores
        )
        
        taxonomy_summary = {
            'dataset': dataset_name,
            'seed': seed,
            'distribution': taxonomy_report['distribution'],
            'transfer_by_type': taxonomy_report['transfer_by_type'],
            'best_transfer': taxonomy_report['best_transfer_type'],
            'worst_transfer': taxonomy_report['worst_transfer_type']
        }
        

        print("    Decomposing teacher knowledge...")
        knowledge_report = create_knowledge_report(
            teacher, X_test, y_test, student_scores
        )
        
        knowledge_summary = {
            'dataset': dataset_name,
            'seed': seed,
            'components': knowledge_report['components'],
            'transfer_analysis': knowledge_report['transfer_analysis'],
            'summary': knowledge_report['summary']
        }
        

        print("    Analyzing failure modes...")
        failure_report = create_failure_analysis_report(
            teacher_scores, student_scores, y_test, X_test,
            student_predict_fn=distiller.predict
        )
        
        failure_summary = {
            'dataset': dataset_name,
            'seed': seed,
            'overview': failure_report['overview'],
            'failure_breakdown': failure_report['failure_breakdown'],
            'severity': failure_report['severity'],
            'recommendations': failure_report['recommendations']
        }
        

        print("    Running diffusion diagnostics...")
        diffusion_report = create_diffusion_diagnostic_report(
            teacher_predict_fn=teacher.predict_scores,
            student_predict_fn=distiller.predict,
            X=X_test,
            y_true=y_test
        )
        
        diffusion_summary = {
            'dataset': dataset_name,
            'seed': seed,
            'teacher_stability': diffusion_report['teacher']['stability'],
            'student_stability': diffusion_report['student']['stability'],
            'comparison': diffusion_report['comparison'],
            'student_findings': diffusion_report['student']['findings'],
            'student_recommendations': diffusion_report['student']['recommendations']
        }
        
        return {
            'performance': performance,
            'taxonomy': taxonomy_summary,
            'knowledge': knowledge_summary,
            'failures': failure_summary,
            'diffusion': diffusion_summary
        }

    def _generate_summary(
        self,
        taxonomy_results: list,
        knowledge_results: list,
        failure_results: list
    ):

        print("\n  Generating summary...")
        
        # Aggregate taxonomy findings
        type_transfer_rates = {}
        for result in taxonomy_results:
            for atype, data in result.get('transfer_by_type', {}).items():
                if atype not in type_transfer_rates:
                    type_transfer_rates[atype] = []
                type_transfer_rates[atype].append(data.get('transfer_rate', 0))
        
        taxonomy_summary = {
            'best_transferring_types': [],
            'worst_transferring_types': [],
            'type_statistics': {}
        }
        
        for atype, rates in type_transfer_rates.items():
            mean_rate = np.mean(rates)
            taxonomy_summary['type_statistics'][atype] = {
                'mean_transfer_rate': mean_rate,
                'std_transfer_rate': np.std(rates),
                'n_observations': len(rates)
            }
            
            if mean_rate > 0.7:
                taxonomy_summary['best_transferring_types'].append(atype)
            elif mean_rate < 0.4:
                taxonomy_summary['worst_transferring_types'].append(atype)
        

        component_retention = {}
        for result in knowledge_results:
            for comp, data in result.get('transfer_analysis', {}).items():
                if comp not in component_retention:
                    component_retention[comp] = []
                component_retention[comp].append(data.get('knowledge_retained', 0))
        
        knowledge_summary = {
            'component_statistics': {}
        }
        
        for comp, rates in component_retention.items():
            knowledge_summary['component_statistics'][comp] = {
                'mean_retention': np.mean(rates),
                'std_retention': np.std(rates)
            }
        

        failure_rates = {
            'collapse': [],
            'hallucination': [],
            'boundary': []
        }
        
        for result in failure_results:
            breakdown = result.get('failure_breakdown', {})
            if 'collapsed_anomalies' in breakdown:
                failure_rates['collapse'].append(breakdown['collapsed_anomalies'].get('rate_of_anomalies', 0))
            if 'hallucinated_anomalies' in breakdown:
                failure_rates['hallucination'].append(breakdown['hallucinated_anomalies'].get('rate_of_normals', 0))
            if 'boundary_confusions' in breakdown:
                failure_rates['boundary'].append(breakdown['boundary_confusions'].get('rate', 0))
        
        failure_summary = {
            'mean_collapse_rate': np.mean(failure_rates['collapse']) if failure_rates['collapse'] else 0,
            'mean_hallucination_rate': np.mean(failure_rates['hallucination']) if failure_rates['hallucination'] else 0,
            'mean_boundary_confusion_rate': np.mean(failure_rates['boundary']) if failure_rates['boundary'] else 0
        }
        

        findings = []
        
        if taxonomy_summary['worst_transferring_types']:
            findings.append(
                f"Anomaly types that transfer poorly: {', '.join(taxonomy_summary['worst_transferring_types'])}. "
                "These may require specialized handling during distillation."
            )
        
        if taxonomy_summary['best_transferring_types']:
            findings.append(
                f"Anomaly types that transfer well: {', '.join(taxonomy_summary['best_transferring_types'])}. "
                "The student successfully learns to detect these."
            )

        if knowledge_summary['component_statistics']:
            best_comp = max(
                knowledge_summary['component_statistics'].items(),
                key=lambda x: x[1]['mean_retention'] if x[0] != 'ensemble' else 0
            )
            worst_comp = min(
                knowledge_summary['component_statistics'].items(),
                key=lambda x: x[1]['mean_retention'] if x[0] != 'ensemble' else 1
            )
            
            findings.append(
                f"Best transferring knowledge: {best_comp[0]} ({best_comp[1]['mean_retention']:.1%} retention). "
                f"Worst: {worst_comp[0]} ({worst_comp[1]['mean_retention']:.1%} retention)."
            )
        
        if failure_summary['mean_collapse_rate'] > 15:
            findings.append(
                f"High anomaly collapse rate ({failure_summary['mean_collapse_rate']:.1f}%). "
                "Student frequently misses anomalies the teacher detected."
            )
        
        self.results['summary'] = {
            'taxonomy': taxonomy_summary,
            'knowledge': knowledge_summary,
            'failures': failure_summary,
            'key_findings': findings,
            'n_datasets_analyzed': len(set(r['dataset'] for r in self.results['performance'])),
            'total_experiments': len(self.results['performance'])
        }

    def run_baseline_comparison(self, datasets: list[str] = None):

        datasets = datasets or MAIN_DATASETS
        
        print("\n" + "=" * 70)
        print("BASELINE COMPARISON")
        print("=" * 70)
        
        for dataset_name in datasets:
            print(f"\n  Dataset: {dataset_name}")
            
            try:
                X_train, X_test, y_train, y_test = load_dataset(dataset_name)
                X_train_normal = X_train[y_train == 0]
                
                if len(X_train_normal) < self.MIN_DATASET_SIZE:
                    continue
                
                contamination = float(y_test.mean())
                
                baseline_results = run_fixed_baseline_comparison(
                    X_train_normal, X_test, y_test, contamination
                )
                
                self.results['baselines'].append({
                    'dataset': dataset_name,
                    'results': baseline_results['baselines'],
                    'summary': baseline_results['summary']
                })
                
                print(f"    Best baseline: {baseline_results['summary'].get('best_method', 'N/A')}")
                
            except Exception as e:
                print(f"    Error: {e}")
        
        self._save_results()

    def print_key_findings(self):

        if not self.results['summary']:
            print("No summary available. Run analysis first.")
            return
        
        print("\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)
        
        for i, finding in enumerate(self.results['summary'].get('key_findings', []), 1):
            print(f"\n{i}. {finding}")
        
        print("\n" + "-" * 70)
        print("RECOMMENDATIONS")
        print("-" * 70)
        

        all_recs = set()
        for failure in self.results.get('failure_analysis', []):
            for rec in failure.get('recommendations', []):
                all_recs.add(rec)
        
        for i, rec in enumerate(all_recs, 1):
            print(f"\n{i}. {rec}")


def main():
    parser = argparse.ArgumentParser(description='Knowledge Transfer Analysis for Anomaly Detection')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'analysis', 'baselines', 'summary'])
    parser.add_argument('--datasets', type=str, nargs='+', default=None)
    parser.add_argument('--n-runs', type=int, default=3)
    parser.add_argument('--output', type=str, default='experiments/knowledge_transfer_results')
    args = parser.parse_args()
    
    runner = KnowledgeTransferExperiment(output_dir=args.output)
    
    if args.mode == 'full':
        runner.run_full_analysis(args.datasets, n_runs=args.n_runs)
        runner.run_baseline_comparison(args.datasets)
        runner.print_key_findings()
    elif args.mode == 'analysis':
        runner.run_full_analysis(args.datasets, n_runs=args.n_runs)
        runner.print_key_findings()
    elif args.mode == 'baselines':
        runner.run_baseline_comparison(args.datasets)
    elif args.mode == 'summary':
        runner.print_key_findings()


if __name__ == '__main__':
    main()
