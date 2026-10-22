"""Render an actual loss CSV as SVG, using only the standard library."""
import argparse
import csv
from html import escape
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv_path', required=True,default="./runs/reranker/loss.csv")
    parser.add_argument('--output_path', required=True, default="./figures/retriever_loss.svg")
    parser.add_argument('--title', default='Training loss')
    args = parser.parse_args()
    with Path(args.csv_path).open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 5:
        raise ValueError('Report Q1 requires at least 5 actual loss data points; run more training steps')
    steps, losses = [float(row['step']) for row in rows], [float(row['loss']) for row in rows]
    low, high = min(losses), max(losses)
    span = max(high - low, 1e-6)
    low -= span * 0.05
    high += span * 0.05
    width, height = 900, 480
    x = lambda value: 85 + (value - steps[0]) / max(1, steps[-1] - steps[0]) * 760
    y = lambda value: 395 - (value - low) / (high - low) * 320
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<g font-family="sans-serif" font-size="14" fill="#26334a">',
                f'<text x="450" y="35" text-anchor="middle" font-size="22">{escape(args.title)}</text>']
    for i in range(6):
        value = low + i * (high - low) / 5
        coordinate = y(value)
        elements.extend([f'<path d="M85 {coordinate:.2f}H845" stroke="#e1e6ed"/>',
                         f'<text x="75" y="{coordinate+5:.2f}" text-anchor="end">{value:.4f}</text>'])
    for i in range(6):
        step = steps[0] + (steps[-1] - steps[0]) * i / 5
        elements.append(f'<text x="{x(step):.2f}" y="420" text-anchor="middle">{step:.0f}</text>')
    points = ' '.join(f'{x(step):.2f},{y(loss):.2f}' for step, loss in zip(steps, losses))
    elements.extend([f'<polyline points="{points}" fill="none" stroke="#1769c2" stroke-width="2"/>',
                     '<text x="450" y="460" text-anchor="middle">Optimizer step</text>',
                     '<text x="20" y="240" text-anchor="middle" transform="rotate(-90 20 240)">Loss</text>',
                     '</g></svg>'])
    
    destination = Path(args.output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(elements) + '\n', encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    main()
