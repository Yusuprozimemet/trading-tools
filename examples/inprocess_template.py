"""
Template for making a Streamlit page that is safe to run in-process by the multi-runner.
Save this pattern as a reference for converting your scripts.
"""
import streamlit as st


def main():
    st.title("In-process friendly example page")
    st.sidebar.header("Example sidebar")
    st.write("This page is safe to run inside the runner's Streamlit session.")


if __name__ == "__main__":
    main()
