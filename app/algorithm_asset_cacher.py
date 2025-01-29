class AlgorithmAssetCacher:
    @classmethod
    def get_user_collection_params(cls, actor_handle, direction):
        return {
            "asset_type": "user_collection",
            "asset_parameters": {"actor_handle": actor_handle, "direction": direction},
            "keyname_template": "{actor_handle}__{direction}",
        }

    @classmethod
    def get_starter_pack_params(cls, starter_pack_url):
        return {
            "asset_type": "starter_pack",
            "asset_parameters": {"starter_pack_url": starter_pack_url},
            "keyname_template": "{starter_pack_url}",
        }

    @classmethod
    def get_list_params(cls, list_url):
        return {
            "asset_type": "user_list",
            "asset_parameters": {"list_url": list_url},
            "keyname_template": "{list_url}",
        }
