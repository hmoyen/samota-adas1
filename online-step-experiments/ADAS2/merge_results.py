import os
import pandas as pd
import click

@click.command()
@click.option('--logdir', default="out", help='Log directory.', type=str)
def main(logdir):
    df_reqs = None
    df_score = None
    for f in os.listdir(logdir):
        if f.startswith("reqs"):
            if df_reqs is None:
                df_reqs = pd.read_csv(os.path.join(logdir, f))
                df_reqs["alg"] = f.split("_")[1]
            else:
                df = pd.read_csv(os.path.join(logdir, f))
                df["alg"] = f.split("_")[1]
                df_reqs = pd.concat([df_reqs, df])
        elif f.startswith("score"):
            if df_score is None:
                df_score = pd.read_csv(os.path.join(logdir, f))
                df_score["alg"] = f.split("_")[1]
            else:
                df = pd.read_csv(os.path.join(logdir, f))
                df["alg"] = f.split("_")[1]
                df_score = pd.concat([df_score, df])
    df_reqs.to_csv(f'{logdir}/reqs_all.csv', index=False)
    df_score.to_csv(f'{logdir}/score_all.csv', index=False)

if __name__ == "__main__":
    main()