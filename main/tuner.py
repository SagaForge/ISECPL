import numpy as np
import pandas as pd
from tsampler import Sampler
import networkx as nx
import matplotlib.pyplot as plt

class TreeNode:
    def __init__(self, feature, best_value, avg_performance, values):
        self.feature = feature
        self.best_value = best_value
        self.avg_performance = avg_performance
        self.values = values
        self.children = []  # Each node will have at least 2 children

    def add_child(self, node):
        self.children.append(node)

class HierarchalManager:
    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.root = None  # The top node of our hierarchical tree

    def construct_tree(self, sampled_df=None):
        if sampled_df is None:
            sampled_df = self.sampler.observed_random_sample()
        
        feature_columns = [col for col in sampled_df.columns if col != self.sampler.performance_col]
        hierarchy = {}
        
        for feature in feature_columns:
            hierarchy[feature] = {}
            for value in sampled_df[feature].unique():
                subgroup = sampled_df[sampled_df[feature] == value]
                avg_performance = subgroup[self.sampler.performance_col].mean()
                hierarchy[feature][value] = {
                    "count": len(subgroup),
                    "avg_performance": avg_performance,
                    "rows": subgroup.drop(columns=[self.sampler.performance_col]).values.tolist()
                }
        
        sorted_features = sorted(hierarchy.keys(), key=lambda f: min(hierarchy[f].values(), key=lambda x: x['avg_performance'])['avg_performance'])
        
        self.root = self._build_recursive_tree(sorted_features, hierarchy)
        print("[INFO] Hierarchical tree constructed successfully.")

    def _build_recursive_tree(self, features, hierarchy, parent_value=None):
        if not features:
            return None

        feature = features[0]
        best_value, best_data = min(hierarchy[feature].items(), key=lambda x: x[1]["avg_performance"])
        node = TreeNode(feature, best_value, best_data["avg_performance"], hierarchy[feature])
        
        remaining_features = features[1:]
        for value, data in hierarchy[feature].items():
            if remaining_features:
                child_node = self._build_recursive_tree(remaining_features, hierarchy, value)
                if child_node:
                    node.add_child(child_node)
        
        return node

    def traverse_tree(self, node=None, level=0):
        if node is None:
            node = self.root
        
        if node is None:
            print("[ERROR] No tree constructed yet.")
            return
        
        print("  " * level + f"Feature: {node.feature}, Best Value: {node.best_value}, Avg Performance: {node.avg_performance}")
        for child in node.children:
            self.traverse_tree(child, level + 1)
    
    def find_best_feature(self):
        return self.root.feature if self.root else None

    def find_worst_feature(self):
        def find_deepest_worst(node):
            if not node.children:
                return node
            return max(node.children, key=lambda n: n.avg_performance, default=node)
        
        return find_deepest_worst(self.root).feature if self.root else None
    
    def update_tree(self, new_sampled_df):
        self.construct_tree(new_sampled_df)

    def visualize_tree(self):
        if not self.root:
            print("[ERROR] No tree to visualize.")
            return
        
        graph = nx.DiGraph()
        
        def add_edges(node, parent=None):
            if node:
                graph.add_node(node.feature, label=f"{node.feature}\n{node.best_value}\n{node.avg_performance:.2f}")
                if parent:
                    graph.add_edge(parent.feature, node.feature)
                for child in node.children:
                    add_edges(child, node)
        
        add_edges(self.root)
        pos = nx.spring_layout(graph)
        labels = nx.get_node_attributes(graph, 'label')
        plt.figure(figsize=(10, 6))
        nx.draw(graph, pos, labels=labels, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000)
        plt.show()

if __name__ == "__main__":
    sampler = Sampler('/home/connor/university/isecpl/temp/LLVM_cleaned.csv', budget=500, inital_sample=0.4, performance_col=None, minimize=True)
    sampler.stage_sampler()
    manager = HierarchalManager(sampler=sampler)
    manager.construct_tree()
    manager.traverse_tree()
    print("Best Feature:", manager.find_best_feature())
    print("Worst Feature:", manager.find_worst_feature())
    manager.visualize_tree()
