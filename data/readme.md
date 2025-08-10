## Overview

This folder contains the initial data of short videos from the TopicVid dataset, including the raw content and the detailed data after re-indexing and processing. We also provide features extracted from different data using models, as well as the constructed heterogeneous network graphs, which can be found at [link](https://huggingface.co/datasets/chensh911/TopicVid).



## Dataset Description

### 1. available_dataset_with_subtopic.json

This file contains the raw data of short video content and interaction statistics.

| Field Name | Type | Description |
|------------|------|-------------|
| `url` | `string` | Direct link to the video on the platform. |
| `desc` | `string` | Description text of the video content. |
| `title` | `string` | Title of the video post. |
| `content` | `string` | Additional text content; may be empty. |
| `user_id` | `string` | Unique identifier of the publishing user. |
| `nickname` | `string` | Display name of the publishing user. |
| `duration` | `integer` | Video duration in seconds. |
| `platform` | `string` | Source platform name (e.g., Douyin, Kuaishou). |
| `post_create_time` | `string` | Time of publication in `"YYYY-MM-DD HH:MM:SS"` format. |
| `topic` | `string` | Main topic associated with the video. |
| `subtopic` | `string` | Numbered subcategory under the main `topic`. |
| `time_frames` | `dict` | Interaction statistics recorded at different dates.<br> - **Key**: Date (`YYYY-MM-DD`)<br> - **Value**: Dictionary with fields:<ul><li>`fans_count` — Number of followers</li><li>`like_count` — Number of likes</li><li>`view_count` — Number of views</li><li>`share_count` — Number of shares</li><li>`collect_count` — Number of collections</li><li>`comment_count` — Number of comments</li></ul> |
| `comments` | `dict` | Collection of user comments.<br> - **Key**: Comment index (string)<br> - **Value**: Dictionary with fields:<ul><li>`comment_user_id` — Commenting user ID</li><li>`comment_nickname` — Commenting user's display name</li><li>`comment_content` — Comment text</li><li>`comment_time` — Time of comment</li><li>`ip_address` — IP location of the commenting user</li></ul> |


### 2. comment.csv
This file contains user comments associated with the short videos.

| Field Name | Type | Description |
|------------|------|-------------|
| `id` | `integer` | Unique identifier for the comment. |
| `text` | `string` | The content of the comment text. |

### 3. video.csv
This file contains basic information for each video node, including its unique ID, source URL, and associated key.

| Field Name | Type | Description |
|------------|------|-------------|
| `id` | `integer` | Unique numerical ID assigned to the video node. |
| `url` | `string` | Source URL of the video. |
| `key` | `string` | Associated key of the video of raw json file (usually same to id) |


### 4. subtopic.csv
This file contains subtopic node information, where each subtopic is assigned a unique ID.

| Field Name | Type | Description |
|------------|------|-------------|
| `id` | `integer` | Unique numerical ID assigned to the subtopic node. |
| `text` | `string` | Parent topic text and this subtopic’s number **within that topic**, joined with an underscore. Example: `Sports_15` denotes the subtopic whose within-topic number is 15 under `Sports`. |


### 5. topic.csv
This file contains topic node information, where each topic is assigned a unique ID.

| Field Name | Type | Description |
|------------|------|-------------|
| `id` | `integer` | Unique numerical ID assigned to the topic node. |
| `text` | `string` | The textual name of the topic. |


### 6. desc.csv
This file contains description text nodes, each with a unique ID.

| Field Name | Type | Description |
|------------|------|-------------|
| `id` | `integer` | Unique numerical ID assigned to the description node. |
| `text` | `string` | The description text associated with the video. |


### 7. title.csv
This file contains title text nodes, each with a unique ID.

| Field Name | Type | Description |
|------------|------|-------------|
| `id` | `integer` | Unique numerical ID assigned to the title node. |
| `text` | `string` | The title text associated with the video. |


### 8. content.csv
This file contains content text nodes, each with a unique ID.

| Field Name | Type | Description |
|------------|------|-------------|
| `id` | `integer` | Unique numerical ID assigned to the content node. |
| `text` | `string` | The content text associated with the video. |



### 9. user.csv
This file contains user node information.

| Field Name | Type | Description |
|------------|------|-------------|
| `id` | `integer` | Unique numerical ID assigned to the user node. |
| `user_id` | `string` | Platform-specific unique identifier for the user. |

---

### 10.fans.csv
This file contains fan count feature vectors for fan nodes.

| Field Name | Type | Description |
|------------|------|-------------|
| `id` | `integer` | Unique numerical ID assigned to the fan node. |
| `feat` | `string` | Feature representation of the fan node (stored as string, may represent serialized vector data). |


### 11. subtopic_label.csv
This file contains the peak popularity labels for each subtopic, where the labels represent the peak values of short video views, comments, and shares.  
All label values have been transformed using the natural logarithm.

| Field Name | Type | Description |
|------------|------|-------------|
| `id` | `integer` | Unique numerical ID assigned to the subtopic node. |
| `view` | `float` | Log-transformed peak view count for videos under this subtopic. |
| `comment` | `float` | Log-transformed peak comment count for videos under this subtopic. |
| `share` | `float` | Log-transformed peak share count for videos under this subtopic. |
