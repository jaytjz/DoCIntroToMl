import matplotlib.pyplot as plt

class Node:
    def __init__(self, attribute, value, left_node, right_node, leaf):
        self.attribute = attribute
        self.value = value
        self.left_node = left_node
        self.right_node = right_node
        self.leaf = leaf 

    def print_tree(self, indent=""):
        if self.leaf is True:
            print(f"{indent}Classification: {self.value}")
        else:
            print(f"{indent}Attribute: {self.attribute}, Value: {self.value}")
            print(f"{indent}--> Left Branch:")
            self.left_node.print_tree(indent + "    ")
            print(f"{indent}--> Right Branch:")
            self.right_node.print_tree(indent + "    ")

    def count_nodes(self):
        if self.leaf:
            return 1
        return 1 + self.left_node.count_nodes() + self.right_node.count_nodes()
    
    def prune_tree(self):
        if not self.leaf:
            self.left_node.prune_tree()
            self.right_node.prune_tree()

            if self.left_node.leaf and self.right_node.leaf:
                if self.left_node.value == self.right_node.value:
                    self.attribute = -1
                    self.value = self.left_node.value
                    self.left_node = None
                    self.right_node = None
                    self.leaf = True
                    print(f"Pruned node")

    def aggressive_prune(self, test_data, root, test_func, print_prunes=False):
        if not self.leaf:
            self.left_node.aggressive_prune(test_data,root,test_func)
            self.right_node.aggressive_prune(test_data,root,test_func)

            old_attribute = self.attribute
            old_value = self.value
            old_left_node = self.left_node
            old_right_node = self.right_node
            old_accuracy = test_func(root,test_data)
            
            def try_prune_to(new_value):
                self.leaf = True
                self.attribute = -1
                self.value = new_value
                self.left_node = None
                self.right_node = None

                new_accuracy = test_func(root, test_data)
                if new_accuracy < old_accuracy:
                    self.leaf = False
                    self.attribute = old_attribute
                    self.value = old_value
                    self.left_node = old_left_node
                    self.right_node = old_right_node
                else:
                    if print_prunes:
                        print("Succesful prune")
            if self.left_node and self.left_node.leaf:
                try_prune_to(self.left_node.value)
            if self.right_node and self.right_node.leaf:
                try_prune_to(self.right_node.value)
                
    def prune_until_converged(self, test_data, root, test_func):
        prev_count = -1
        while True:
            current_count = self.count_nodes()
            if current_count == prev_count:
                break
            prev_count = current_count
            self.aggressive_prune(test_data, root, test_func)

    def traverse_tree(self, instance):
        if self.leaf:
            return self.value
        if(instance[self.attribute] < self.value):
            return self.left_node.traverse_tree(instance)
        else:
            return self.right_node.traverse_tree(instance)
    
    def compute_positions(self, depth=0, positions=None):
        if positions is None:
            positions = {}

        if self.left_node and self.right_node:
            left_positions = self.left_node.compute_positions(depth + 1, positions)
            right_positions = self.right_node.compute_positions(depth + 1, positions)

            left_x = positions[self.left_node][0]
            right_x = positions[self.right_node][0]

            x = (left_x + right_x) / 2

        elif self.left_node:
            self.left_node.compute_positions(depth + 1, positions)
            x = positions[self.left_node][0]

        elif self.right_node:
            self.right_node.compute_positions(depth + 1, positions)
            x = positions[self.right_node][0]

        else:
            leaf_x = sum(1 for node, (xx, yy) in positions.items() if node.leaf)
            x = leaf_x

        y = depth
        positions[self] = (x, y)
        return positions

    def draw_tree(self):
        positions = self.compute_positions()

        max_x = max(x for x, y in positions.values())
        max_y = max(y for x, y in positions.values())
        fig_width = min(12, (max_x + 2) * 1.5)
        fig_height = min(8, (max_y + 2) * 2.0)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_aspect('equal')
        ax.axis('off')

        scale_x = 2.0 
        scale_y = 3.0
        ax.invert_yaxis()

        def draw_edges(node):
            if node.left_node:
                x1, y1 = positions[node]
                x2, y2 = positions[node.left_node]
                ax.plot([x1 * scale_x, x2 * scale_x], [y1 * scale_y, y2 * scale_y], 'k-', zorder=1)
                draw_edges(node.left_node)
            if node.right_node:
                x1, y1 = positions[node]
                x2, y2 = positions[node.right_node]
                ax.plot([x1 * scale_x, x2 * scale_x], [y1 * scale_y, y2 * scale_y], 'k-', zorder=1)
                draw_edges(node.right_node)

        draw_edges(self)

        for n, (x, y) in positions.items():
            x *= scale_x
            y *= scale_y

            circle = plt.Circle((x, y), 0.6, color='skyblue', ec='black', zorder=2)
            ax.add_patch(circle)

            if n.leaf:
                label = f"Class:\n{n.value}"
            else:
                label = f"Attr {n.attribute}\n< {n.value:.1f}"

            ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)

        plt.tight_layout()
        plt.show()



