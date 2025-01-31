import click
import os
import subprocess


@click.command()
@click.argument("dataset_name", type=str, default="hoge")
def main(dataset_name):
    if not os.path.exists(f"{os.environ['SGE_LOCALDIR']}/datasets/{dataset_name}/"):
        print("Copy dataaset to SGE_LOCALDIR !!")
        os.makedirs(f"{os.environ['SGE_LOCALDIR']}/datasets/", exist_ok=True)
        copy_dataset_path = f"./datasets/{dataset_name}.tar.gz"

        # Copy dataset to local storage
        subprocess.run(
            [
                "cp",
                copy_dataset_path,
                f"{os.environ['SGE_LOCALDIR']}/datasets/",
            ]
        )
        # extract dataset images
        subprocess.run(
            [f"tar -I  pigz -xf {dataset_name}.tar.gz"],
            cwd=f"{os.environ['SGE_LOCALDIR']}/datasets/",
            shell=True,
        )


if __name__ == "__main__":
    main()
