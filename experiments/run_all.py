"""Run every experiment in order (about 10-15 minutes on a laptop)."""
import common  # noqa: F401
import e0_tune_deepc
import e1_identification
import e2_closed_loop
import e3_figures

if __name__ == "__main__":
    e0_tune_deepc.main()
    e1_identification.main()
    e2_closed_loop.main()
    e3_figures.main()
