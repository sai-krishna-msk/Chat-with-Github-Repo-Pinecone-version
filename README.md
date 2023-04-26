## Chat-with-Github-Repo-Pinecone-version

- This Repository is a version of [Chat-with-github-Repo](https://github.com/peterw/Chat-with-Github-Repo), but uses pinecone instead of deeplake.

- Please visit the original project for instructions on how to use the project

- The only changes are instead of deeplake you will be using pinecone as your vectordatabase
	- Get your pinecone API	key and region to fill in the .env.example, instead of deeplake keys and its info 
	- Fill in the name you want, your pinecone index(~database name) to be in "INDEX_NAME" field in .env.example
	- if it's a free Pinecone account make sure to delete the existing one before you run the code, as the free version of Pinecone only allows one index.

# Credits
[Peter](https://github.com/peterw)


