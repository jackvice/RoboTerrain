// src/actor_pos_publisher.cc
#include <chrono>
#include <optional>
#include <string>

#include <sdf/Element.hh>
#include <ignition/msgs/pose.pb.h>
#include <ignition/msgs/Utility.hh>
#include <ignition/transport/Node.hh>

#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/Actor.hh>
#include <ignition/gazebo/EntityComponentManager.hh>
#include <ignition/gazebo/EventManager.hh>
#include <ignition/gazebo/components/Name.hh>

// Defines IGNITION_ADD_PLUGIN / IGNITION_ADD_PLUGIN_ALIAS
#include <ignition/plugin/Register.hh>

namespace ignition {
namespace gazebo {
namespace systems {

class ActorPosePublisher final
  : public System,
    public ISystemConfigure,
    public ISystemPreUpdate {
public:

  void Configure(const Entity& entity,
                 const std::shared_ptr<const sdf::Element> &sdf,
                 EntityComponentManager&,
                 EventManager&) override
  {
    // Store the entity directly - no need to search!
    actorEntity_ = entity;
  
    if (sdf) {
      if (sdf->HasElement("topic"))
        topic_ = sdf->Get<std::string>("topic");
      if (sdf->HasElement("rate_hz"))
        rateHz_ = sdf->Get<double>("rate_hz");
    }
    pub_ = node_.Advertise<msgs::Pose>(topic_);
  }

  void PreUpdate(const UpdateInfo &info, EntityComponentManager &ecm) override
  {
    if (info.paused) return;

    // ~30 Hz throttle (sim time)
    const auto now =
      std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime);
    const auto period =
      std::chrono::nanoseconds(static_cast<int64_t>(1e9 / rateHz_));
    if (lastEmit_ != std::chrono::nanoseconds::zero() &&
        (now - lastEmit_) < period)
      return;

    Actor a(actorEntity_);
    if (!a.Valid(ecm)) return;

    // WorldPose returns std::optional<Pose3d>
    const auto pOpt = a.WorldPose(ecm);
    if (!pOpt.has_value()) return;
    const auto &p = *pOpt;

    msgs::Pose msg;
    msgs::Set(msg.mutable_position(),    p.Pos());
    msgs::Set(msg.mutable_orientation(), p.Rot());
    pub_.Publish(msg);

    lastEmit_ = now;
  }

private:
  std::string actorName_{"linear_actor"};
  std::string topic_{"/linear_actor/pose"};
  double rateHz_{30.0};

  Entity actorEntity_{kNullEntity};
  transport::Node node_;
  transport::Node::Publisher pub_;
  std::chrono::nanoseconds lastEmit_{};
};

}  // namespace systems
}  // namespace gazebo
}  // namespace ignition

// Register + give an explicit alias that matches SDF
IGNITION_ADD_PLUGIN(ignition::gazebo::systems::ActorPosePublisher,
  ignition::gazebo::System,
  ignition::gazebo::ISystemConfigure,
  ignition::gazebo::ISystemPreUpdate)

//IGNITION_ADD_PLUGIN_ALIAS(ignition::gazebo::systems::ActorPosePublisher,
//  "ignition::gazebo::systems::ActorPosePublisher")

IGNITION_ADD_PLUGIN_ALIAS(ignition::gazebo::systems::ActorPosePublisher,
  "ActorPosePublisher")
