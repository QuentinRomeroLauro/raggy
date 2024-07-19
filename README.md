# Bootstrap Evaluations for LLM/RAG Pipelines

## Running the Project

### Launch the debugging interface
```
cd interfaces/front-end
npm start
```
### Launch the Flask app
From `/interfaces_for_rag`:
```
python interfaces/streamToInterface.py
```

### Edit and run your pipeline
From `/interfaces_for_rag`:
```
python task/pipeline.py
```
```
interfaces_for_rag
├─ .git
│  ├─ HEAD
│  ├─ config
│  ├─ description
│  ├─ hooks
│  │  ├─ applypatch-msg.sample
│  │  ├─ commit-msg.sample
│  │  ├─ fsmonitor-watchman.sample
│  │  ├─ post-update.sample
│  │  ├─ pre-applypatch.sample
│  │  ├─ pre-commit.sample
│  │  ├─ pre-merge-commit.sample
│  │  ├─ pre-push.sample
│  │  ├─ pre-rebase.sample
│  │  ├─ pre-receive.sample
│  │  ├─ prepare-commit-msg.sample
│  │  ├─ push-to-checkout.sample
│  │  └─ update.sample
│  ├─ info
│  │  └─ exclude
│  ├─ objects
│  │  ├─ info
│  │  └─ pack
│  └─ refs
│     ├─ heads
│     └─ tags
├─ .gitignore
├─ README.md
├─ __init__.py
├─ counter.txt
├─ interfaces
│  ├─ __init__.py
│  ├─ __pycache__
│  │  ├─ __init__.cpython-312.pyc
│  │  └─ run.cpython-312.pyc
│  ├─ front-end
│  │  ├─ .git
│  │  │  ├─ COMMIT_EDITMSG
│  │  │  ├─ HEAD
│  │  │  ├─ config
│  │  │  ├─ description
│  │  │  ├─ hooks
│  │  │  │  ├─ applypatch-msg.sample
│  │  │  │  ├─ commit-msg.sample
│  │  │  │  ├─ fsmonitor-watchman.sample
│  │  │  │  ├─ post-update.sample
│  │  │  │  ├─ pre-applypatch.sample
│  │  │  │  ├─ pre-commit.sample
│  │  │  │  ├─ pre-merge-commit.sample
│  │  │  │  ├─ pre-push.sample
│  │  │  │  ├─ pre-rebase.sample
│  │  │  │  ├─ pre-receive.sample
│  │  │  │  ├─ prepare-commit-msg.sample
│  │  │  │  ├─ push-to-checkout.sample
│  │  │  │  └─ update.sample
│  │  │  ├─ index
│  │  │  ├─ info
│  │  │  │  └─ exclude
│  │  │  ├─ logs
│  │  │  │  ├─ HEAD
│  │  │  │  └─ refs
│  │  │  │     └─ heads
│  │  │  │        └─ main
│  │  │  ├─ objects
│  │  │  │  ├─ 08
│  │  │  │  │  └─ 0d6c77ac21bb2ef88a6992b2b73ad93daaca92
│  │  │  │  ├─ 1f
│  │  │  │  │  └─ 03afeece5ac28064fa3c73a29215037465f789
│  │  │  │  ├─ 20
│  │  │  │  │  └─ a994b2359d781406e3eaf34beb3dc90082bcef
│  │  │  │  ├─ 37
│  │  │  │  │  └─ 845757234ccb68531c10cf7a2ffc589c47e342
│  │  │  │  ├─ 4d
│  │  │  │  │  └─ 29575de80483b005c29bfcac5061cd2f45313e
│  │  │  │  ├─ 52
│  │  │  │  │  └─ 53d3ad9e6be6690549cb255f5952337b02401d
│  │  │  │  ├─ 58
│  │  │  │  │  └─ beeaccd87e230076cab531b8f418f40b6d1aeb
│  │  │  │  ├─ 74
│  │  │  │  │  └─ b5e053450a48a6bdb4d71aad648e7af821975c
│  │  │  │  ├─ 7b
│  │  │  │  │  └─ 1e172334cdc980bee9ad6a8a87cf21e0ed18d8
│  │  │  │  ├─ 82
│  │  │  │  │  └─ 149d7434ba214b401305354e0b04f7f0e7f02b
│  │  │  │  ├─ 8e
│  │  │  │  │  └─ 29b36dea7f04ae8729d8b33ecc05c3c9b0fe46
│  │  │  │  ├─ 8f
│  │  │  │  │  └─ 2609b7b3e0e3897ab3bcaad13caf6876e48699
│  │  │  │  ├─ 9d
│  │  │  │  │  └─ fc1c058cebbef8b891c5062be6f31033d7d186
│  │  │  │  ├─ a1
│  │  │  │  │  └─ 1777cc471a4344702741ab1c8a588998b1311a
│  │  │  │  ├─ a4
│  │  │  │  │  └─ e47a6545bc15971f8f63fba70e4013df88a664
│  │  │  │  ├─ aa
│  │  │  │  │  └─ 069f27cbd9d53394428171c3989fd03db73c76
│  │  │  │  ├─ d2
│  │  │  │  │  └─ af4247cb75a9772f9cab41f8888b8b0c1b5370
│  │  │  │  ├─ d5
│  │  │  │  │  └─ 63c0fb10ba0e42724b21286eb546ee4e5734fc
│  │  │  │  ├─ da
│  │  │  │  │  └─ e5e7c5915ff3a1f0d77be175b400f71098af78
│  │  │  │  ├─ e9
│  │  │  │  │  └─ e57dc4d41b9b46e05112e9f45b7ea6ac0ba15e
│  │  │  │  ├─ ec
│  │  │  │  │  └─ 2585e8c0bb8188184ed1e0703c4c8f2a8419b0
│  │  │  │  ├─ fc
│  │  │  │  │  └─ 44b0a3796c0e0a64c3d858ca038bd4570465d9
│  │  │  │  ├─ info
│  │  │  │  └─ pack
│  │  │  └─ refs
│  │  │     ├─ heads
│  │  │     │  └─ main
│  │  │     └─ tags
│  │  ├─ .gitignore
│  │  ├─ README.md
│  │  ├─ node_modules
└─ task
   ├─ __init__.py
   ├─ __pycache__
   │  ├─ __init__.cpython-312.pyc
   │  └─ pipeline.cpython-312.pyc
   ├─ documents
   │  └─ QuentinRomeroLauro-SWE-Resume-24.pdf
   ├─ load_documents.py
   ├─ pipeline.py
   └─ store
      ├─ chroma_10000_1
      │  ├─ 9aae6b72-5c30-4999-9997-344040580331
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      ├─ chroma_1000_1
      │  ├─ b49ef767-530c-4771-b8b3-55d359783c7c
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      ├─ chroma_1000_10
      │  ├─ 49221862-bc0b-4597-a3c0-59c548def84a
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      ├─ chroma_100_0
      │  ├─ 2cbac5d3-9bee-4c19-9c17-411d21def9e4
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      ├─ chroma_100_1
      │  ├─ 6b5cc789-80cb-4220-ad17-29970315abe5
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      ├─ chroma_100_10
      │  ├─ 7a877ad8-2caf-44b4-a5b2-42819e138ea1
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      ├─ chroma_10_0
      │  ├─ c20bb13b-91cc-4934-a2d2-bfdfb3fd4c98
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      ├─ chroma_10_1
      │  ├─ 12448e4b-819d-4a31-86b9-7da4d8c18ab5
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      ├─ chroma_10_10
      │  ├─ chroma.sqlite3
      │  └─ ef65ca6e-0985-4516-9f13-5b16be2159d1
      │     ├─ data_level0.bin
      │     ├─ header.bin
      │     ├─ length.bin
      │     └─ link_lists.bin
      ├─ chroma_120_10
      │  ├─ 3d6ff86c-c5a9-4fb7-b94d-e1b5b1aabe40
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      ├─ chroma_150_10
      │  ├─ 46408150-e4f0-4377-bcfc-9c8818d02428
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      ├─ chroma_250_10
      │  ├─ bc8cbafe-dbf0-4a3f-a785-a6b3e111fcd9
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      ├─ chroma_25_0
      │  ├─ 11220548-9eb3-44d2-b758-f1659d4c9719
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      ├─ chroma_25_1
      │  ├─ 0f1390bd-fb8c-4cd7-9cab-d8216e166d7a
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      ├─ chroma_500_10
      │  ├─ chroma.sqlite3
      │  └─ ea3f0fce-395d-4985-98a4-80ec4d9bf06a
      │     ├─ data_level0.bin
      │     ├─ header.bin
      │     ├─ length.bin
      │     └─ link_lists.bin
      ├─ chroma_50_0
      │  ├─ 03db1093-759a-4c86-ada4-25470c1c3609
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      ├─ chroma_50_1
      │  ├─ chroma.sqlite3
      │  └─ ec696a3d-faaf-425f-940e-96eb058283c4
      │     ├─ data_level0.bin
      │     ├─ header.bin
      │     ├─ length.bin
      │     └─ link_lists.bin
      ├─ chroma_50_10
      │  ├─ 9f659e4d-1d09-482d-811c-6526b0cf41dd
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      ├─ chroma_68_10
      │  ├─ c1c1933f-fae8-463d-8840-541c787bcf90
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      ├─ chroma_70_10
      │  ├─ 3b90773d-dbd5-495d-a59f-1f351ea01d5d
      │  │  ├─ data_level0.bin
      │  │  ├─ header.bin
      │  │  ├─ length.bin
      │  │  └─ link_lists.bin
      │  └─ chroma.sqlite3
      └─ chroma_75_1
         ├─ 2104b3dd-de7a-4618-8d8b-dae1c43554a1
         │  ├─ data_level0.bin
         │  ├─ header.bin
         │  ├─ length.bin
         │  └─ link_lists.bin
         └─ chroma.sqlite3

```